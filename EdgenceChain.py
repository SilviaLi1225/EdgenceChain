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






    def locate_block(self, block_hash: str, chain: BlockChain=None) -> (Block, int, int):
        with self.chain_lock:
            chains = [chain] if chain else [self.active_chain, *self.side_branches]

            for chain_idx, chain in enumerate(chains):
                for height, block in enumerate(chain.chain, 1):
                    if block.id == block_hash:
                        return (block, height, chain_idx)
            return (None, None, None)

    # Proof of work
    def get_next_work_required(self, prev_block_hash: str) -> int:

        """
        Based on the chain, return the number of difficulty bits the next block
        must solve.
        """
        if not prev_block_hash:
            return Params.INITIAL_DIFFICULTY_BITS

        (prev_block, prev_height, pre_chain_idx) = self.locate_block(prev_block_hash)

        if pre_chain_idx != 0:
            return None

        if prev_height % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:
            return prev_block.bits

        with self.chain_lock:
            # #realname CalculateNextWorkRequired
            period_start_block = self.active_chain.chain[max(
                prev_height - Params.DIFFICULTY_PERIOD_IN_BLOCKS, 0)]

        actual_time_taken = prev_block.timestamp - period_start_block.timestamp

        if actual_time_taken < Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
            # Increase the difficulty
            return prev_block.bits + 1
        elif actual_time_taken > Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
            return prev_block.bits - 1
        else:
            # Wow, that's unlikely.
            return prev_block.bits

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
            bits=self.get_next_work_required(prev_block_hash),
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

    def check_block_place(self, block: Block) -> int:

        with self.chain_lock:
            if self.locate_block(block.id)[0]:
                logger.debug(f'mined block {block.id} but already seen, impossible')
                return None

            try:
                chain_idx = block.validate_block(self.active_chain, self.side_branches, self.chain_lock)
            except BlockValidationError as e:
                logger.exception('a mined block %s but failed validation', block.id)
                if e.to_orphan:
                    logger.info(f"mined an orphan block {block.id}, just discard it and go")
                return None

            # If `validate_block()` returned a non-existent chain index, we're
            # creating a new side branch.
            if chain_idx != Params.ACTIVE_CHAIN_IDX and len(self.side_branches) < chain_idx:
                logger.info(
                    f'creating a new side branch (idx={chain_idx}) '
                    f'for block {block.id}')
                self.side_branches.append(BlockChain(idx = chain_idx, chain = []))

            return chain_idx

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
                block = self.assemble_and_solve_block()

                if block:
                    with self.chain_lock:
                        chain_idx  = self.check_block_place(block)

                        if chain_idx is not None:
                            if chain_idx == Params.ACTIVE_CHAIN_IDX:
                                if self.active_chain.connect_block(block, self.active_chain, self.side_branches, \
                                                        self.mempool, self.utxo_set, self.mine_interrupt, self.peers):
                                    with self.chain_lock:
                                        Persistence.save_to_disk(self.active_chain)
                            else:
                                self.side_branches[chain_idx-1].chain.append(block)

                            for _peer in self.peers:
                                Utils.send_to_peer(Message(Actions.BlockRev, block, Params.PORT_CURRENT), _peer)

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



