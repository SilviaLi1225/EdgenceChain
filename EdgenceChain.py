import binascii
import time
import json
import hashlib
import threading
import _thread
import logging
import socketserver
import socket
import random
import os
from functools import lru_cache, wraps
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)


from utils.Errors import BlockValidationError
from utils.Utils import Utils

from params.Params import Params


from ds.Transaction import Transaction
from ds.Block import Block
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain

from persistence import Persistence

from wallet.Wallet import Wallet

from p2p.Message import Message
from p2p.Message import Actions
from p2p.Peer import Peer
from p2p.TCPserver import (ThreadedTCPServer, TCPHandler)


from consensus.Consensus import PoW




logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class EdgenceChain(object):


    def __init__(self):

        self.active_chain: BlockChain = BlockChain(idx=Params.ACTIVE_CHAIN_IDX, chain=[Block.genesis_block()])
        self.side_branches: Iterable[BlockChain] = []
        self.orphan_blocks: Iterable[Block] = []
        self.utxo_set: UTXO_Set = UTXO_Set()
        self.mempool: MemPool = MemPool()
        self.wallet: Wallet = Wallet.init_wallet(Params.WALLET_FILE)
        self.peers: Iterable[Peer] = Peer.init_peers(Params.PEERS_FILE)

        self.mine_interrupt: threading.Event = threading.Event()
        self.ibd_done: threading.Event = threading.Event()
        self.chain_lock: _thread.RLock = threading.RLock()





    def assemble_and_solve_block(self, txns=None)->Block:
        """
        Construct a Block by pulling transactions from the mempool, then mine it.
        """
        with self.chain_lock:
            prev_block_hash = self.active_chain.chain[-1].id if self.active_chain.chain else None

        block = Block(
            version=0,
            prev_block_hash=prev_block_hash,
            merkle_hash='',
            timestamp=int(time.time()),
            bits= Block.get_next_work_required(prev_block_hash, self.active_chain, self.side_branches),
            nonce=0,
            txns=txns or [],
        )

        if block.bits is None:
            return None

        if not block.txns:
            block = self.mempool.select_from_mempool(block, self.utxo_set)

        fees = block.calculate_fees(self.utxo_set)
        my_address = self.wallet()[2]
        coinbase_txn = Transaction.create_coinbase(
            my_address,
            Block.get_block_subsidy(self.active_chain) + fees,
            self.active_chain.height)
        block = block._replace(txns=[coinbase_txn, *block.txns])
        block = block._replace(merkle_hash=MerkleNode.get_merkle_root_of_txns(block.txns).val)

        if len(Utils.serialize(block)) > Params.MAX_BLOCK_SERIALIZED_SIZE:
            raise ValueError('txns specified create a block too large')

        block = PoW.mine(block, self.mine_interrupt)

        return block



    def initial_block_download(self):
        self.ibd_done.clear()
        if self.peers:
            logger.info(f'start initial block download from {len(self.peers)} peers')
            peer_sample = random.sample(self.peers, len(self.peers))
            for peer in peer_sample:
                if not Utils.send_to_peer(Message(Actions.BlocksSyncReq, self.active_chain.chain[-1].id, \
                                              Params.PORT_CURRENT), peer):
                    self.peers.remove(peer)
                    Peer.save_peers(self.peers)
                    logger.info(f'remove dead peer {peer}')
        else:
            self.ibd_done.set()



    def start(self):

        def start_worker(workers, worker):
            workers.append(threading.Thread(target=worker, daemon=True))
            workers[-1].start()

        def mine_forever():
            logger.info(f'thread for mining is started....')
            while True:
                with self.chain_lock:
                    block = self.assemble_and_solve_block()

                if block:
                    for _peer in self.peers:
                        Utils.send_to_peer(Message(Actions.BlockRev, block, Params.PORT_CURRENT), _peer)
                    with self.chain_lock:
                        chain_idx  = TCPHandler.check_block_place(block, self.active_chain, self.utxo_set, \
                                                                  self.mempool, self.side_branches)

                        if chain_idx is not None and chain_idx >= 0:
                            TCPHandler.do_connect_block_and_after(block, chain_idx, self.active_chain, \
                                                                  self.side_branches, self.mempool, \
                                                           self.utxo_set, self.mine_interrupt, self.peers)
                        elif chain_idx is None:
                            logger.info(f'mined already seen block {block.id}, just discard it and go')
                        elif chain_idx == -2:
                            logger.info(f"mined an orphan block {block.id}, just discard it and go")
                        elif chain_idx == -1:
                            logger.exception(f'a mined block {block.id} but failed validation')
                        else:
                            logger.exception(f'unwanted result of check block place')



        # single thread mode, no need for thread lock
        Persistence.load_from_disk(self.active_chain, self.utxo_set)

        workers = []

        server = ThreadedTCPServer(('0.0.0.0', Params.PORT_CURRENT), TCPHandler, self.active_chain, self.side_branches,\
                                   self.orphan_blocks, self.utxo_set, self.mempool, self.peers, self.mine_interrupt, \
                                              self.ibd_done, self.chain_lock)
        start_worker(workers, server.serve_forever)


        self.initial_block_download()
        old_height = self.active_chain.height
        new_height = old_height
        while new_height > old_height:
            old_height = new_height
            wait_times = 3
            while not self.ibd_done.is_set():
                time.sleep(10)
                wait_times -= 1
                if wait_times <= 0:
                    break
            new_height = self.active_chain.height
            logger.info(f'{new_height-old_height} more blocks got this time, waiting for blocks syncing ...')
        self.ibd_done.set()


        start_worker(workers, mine_forever)
        
        [w.join() for w in workers]



