import binascii
import time
import json
import hashlib
import threading
import logging
import socketserver
import socket
import random
import os
from functools import lru_cache, wraps
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)


from ds.UnspentTxOut import UnspentTxOut
from ds.OutPoint import OutPoint
from ds.Transaction import Transaction
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from utils import Utils
from params.Params import Params
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain
from ds.Block import Block

from persistence import Persistence
from wallet.Wallet import Wallet
from p2p.Peer import Peer
import ecdsa
from base58 import b58encode_check
from p2p.P2P import (ThreadedTCPServer, TCPHandler)

from _thread import RLock
from consensus.Consensus import PoW

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class EdgenceChain(object):


    def __init__(self):

        self.active_chain: BlockChain = BlockChain(idx=Params.ACTIVE_CHAIN_IDX, chain=[Params.genesis_block])
        self.side_branches: Iterable[BlockChain] = []
        self.orphan_blocks: Iterable[Block] = []
        self.utxo_set: UTXO_Set = UTXO_Set()
        self.mempool: MemPool = MemPool()
        self.wallet = Wallet.init_wallet(Params.WALLET_FILE)
        self.peers = Peer.init_peers(Params.PEERS_FILE)

        self.mine_interrupt = threading.Event()
        self.ibd_done = threading.Event()
        self.chain_lock = threading.RLock()


        Persistence.load_from_disk(self.active_chain, self.utxo_set, Params.CHAIN_FILE)



    def locate_block(self, block_hash: str, chain=None) -> (Block, int, int):
        with self.chain_lock:
            chains = [chain] if chain else [self.active_chain, *self.side_branches]

            for chain_idx, chain in enumerate(chains):
                for height, block in enumerate(chain):
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

        (prev_block, prev_height, _) = self.locate_block(prev_block_hash)

        if (prev_height + 1) % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:
            return prev_block.bits

        with self.chain_lock:
            # #realname CalculateNextWorkRequired
            period_start_block = self.active_chain[max(
                prev_height - (Params.DIFFICULTY_PERIOD_IN_BLOCKS - 1), 0)]

        actual_time_taken = prev_block.timestamp - period_start_block.timestamp

        if actual_time_taken < Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
            # Increase the difficulty
            return prev_block.bits + 1
        elif actual_time_taken > Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
            return prev_block.bits - 1
        else:
            # Wow, that's unlikely.
            return prev_block.bits

    def assemble_and_solve_block(self, txns=None):
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

        if not block.txns:
            block = self.mempool.select_from_mempool(block, self.utxo_set)

        fees = block.calculate_fees(self.utxo_set)
        my_address = self.wallet.get()[2]
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
        if self.locate_block(block.id)[0]:
            logger.debug(f'ignore block already seen: {block.id}')
            return None

        try:
            chain_idx = block.validate_block(self.active_chain, self.side_branches, self.chain_lock)
        except BlockValidationError as e:
            logger.exception('block %s failed validation', block.id)
            if e.to_orphan:
                logger.info(f"saw orphan block {block.id}")
                self.orphan_blocks.append(e.to_orphan)
            return None

        # If `validate_block()` returned a non-existent chain index, we're
        # creating a new side branch.
        if chain_idx != Params.ACTIVE_CHAIN_IDX and len(self.side_branches) < chain_idx:
            logger.info(
                f'creating a new side branch (idx {chain_idx}) '
                f'for block {block.id}')
            self.side_branches.append([])

        logger.info(f'connecting block {block.id} to chain {chain_idx}')
        return chain_idx



    def start(self):

        def start_worker(workers, worker):
            workers.append(threading.Thread(target=worker, daemon=True))
            workers[-1].start()

        def mine_forever():
            while True:
                block = self.assemble_and_solve_block(self.wallet()[2])

                if block:
                    with self.chain_lock:
                        idx  = self.check_block_place(block)
                        if idx == Params.ACTIVE_CHAIN_IDX:
                            self.active_chain.connect_block(block)
                            Persistence.save_to_disk(self.active_chain)
                        else:
                            self.side_branches[idx-1].chain.append(block)

        workers = []
        server = ThreadedTCPServer(('0.0.0.0', Params.PORT_CURRENT), TCPHandler)
        logger.info(f'[p2p] listening on {Params.PORT_CURRENT}')
        start_worker(workers, server.serve_forever)

        if self.peers:
            logger.info(
                f'start initial block download from {len(self.peers)} peers')
            peer = random.choice(list(self.peers))
            Utils.send_to_peer(GetBlocksMsg(self.active_chain[-1].id), peer)
            self.ibd_done.wait(300.)

        start_worker(mine_forever)
        
        [w.join() for w in workers]



