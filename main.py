#!/usr/bin/env python3
"""
⛼  edgencechain

  putting the rough in "rough consensus"


"""
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

from p2p.P2P import (GetBlocksMsg, InvMsg, ThreadedTCPServer, TCPHandler)
from p2p.Peer import Peer
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain

import ecdsa
from base58 import b58encode_check
from utils import Utils
from wallet import Wallet

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)






from ds.Block import Block
from ds.OutPoint import OutPoint
from ds.TxIn import TxIn
from ds.TxOut import TxOut
from ds.UnspentTxOut import UnspentTxOut
from ds.Transaction import Transaction


from ds.MerkleNode import MerkleNode
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from params.Params import Params


from utils.Utils import Utils


#
# #realname chainActive
active_chain: Iterable[Block] = [Params.genesis_block]

# Branches off of the main chain.
side_branches: Iterable[Iterable[Block]] = []

# Synchronize access to the active chain and side branches.
chain_lock = threading.RLock()





orphan_blocks: Iterable[Block] = []

# Signal to communicate to the mining thread that it should stop mining because
# we've updated the chain with a new block.

# Signal when the initial block download has completed.
ibd_done = threading.Event()


@Utils.with_lock(chain_lock)
def locate_block(block_hash: str, chain=None) -> (Block, int, int):
    chains = [chain] if chain else [active_chain, *side_branches]

    for chain_idx, chain in enumerate(chains):
        for height, block in enumerate(chain):
            if block.id == block_hash:
                return (block, height, chain_idx)
    return (None, None, None)

# Proof of work
def get_next_work_required(prev_block_hash: str) -> int:
    """
    Based on the chain, return the number of difficulty bits the next block
    must solve.
    """
    if not prev_block_hash:
        return Params.INITIAL_DIFFICULTY_BITS

    (prev_block, prev_height, _) = locate_block(prev_block_hash)

    if (prev_height + 1) % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:
        return prev_block.bits

    with chain_lock:
        # #realname CalculateNextWorkRequired
        period_start_block = active_chain[max(
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

def assemble_and_solve_block(mempool: MemPool, active_chain: BlockChain, wallet: Wallet, utxo_set:UTXO_Set, txns=None):
    """
    Construct a Block by pulling transactions from the mempool, then mine it.
    """
    with chain_lock:
        prev_block_hash = active_chain[-1].id if active_chain else None

    block = Block(
        version=0,
        prev_block_hash=prev_block_hash,
        merkle_hash='',
        timestamp=int(time.time()),
        bits=get_next_work_required(prev_block_hash),
        nonce=0,
        txns=txns or [],
    )

    if not block.txns:
        block = mempool.select_from_mempool(block, utxo_set)

    fees = block.calculate_fees(utxo_set)
    my_address = wallet.get()[2]
    coinbase_txn = Transaction.create_coinbase(
        my_address, (Block.get_block_subsidy(active_chain) + fees), active_chain.height)
    block = block._replace(txns=[coinbase_txn, *block.txns])
    block = block._replace(merkle_hash=MerkleNode.get_merkle_root_of_txns(block.txns).val)

    if len(Utils.serialize(block)) > Params.MAX_BLOCK_SERIALIZED_SIZE:
        raise ValueError('txns specified create a block too large')

    return mine(block)

def mine(block: Block, mine_interrupt: threading.Event) -> Block:
    start = time.time()
    nonce = 0
    target = (1 << (256 - block.bits))
    mine_interrupt.clear()

    while int(Utils.sha256d(block.header(nonce)), 16) >= target:
        nonce += 1

        if nonce % 10000 == 0 and mine_interrupt.is_set():
            logger.info('[mining] interrupted')
            mine_interrupt.clear()
            return None

    block = block._replace(nonce=nonce)
    duration = int(time.time() - start) or 0.001
    khs = (block.nonce // duration) // 1000
    logger.info(
        f'[mining] block found! {duration} s - {khs} KH/s - {block.id}')

    return block

def mine_forever():
    while True:
        my_address = Wallet.init_wallet()[2]
        block = assemble_and_solve_block(my_address)

        if block:
            connect_block(block)
            save_to_disk()


def main():
    load_from_disk()
    wallet = Wallet.init_wallet(WALLET_PATH)
    peers = Peer.init_peers()
    utxo_set = UTXO_Set()   #utxo_set.get()
    mempool = MemPool()

    workers = []
    server = ThreadedTCPServer(('0.0.0.0', PORT), TCPHandler)

    def start_worker(fnc):
        workers.append(threading.Thread(target=fnc, daemon=True))
        workers[-1].start()

    logger.info(f'[p2p] listening on {PORT}')
    start_worker(server.serve_forever)

    if peers:
        logger.info(
            f'start initial block download from {len(peers)} peers')
        peer = random.choice(list(peers))
        Utils.send_to_peer(GetBlocksMsg(active_chain[-1].id), peer)
        ibd_done.wait(60.)  # Wait a maximum of 60 seconds for IBD to complete.

    start_worker(mine_forever)
    [w.join() for w in workers]


if __name__ == '__main__':

    main()
