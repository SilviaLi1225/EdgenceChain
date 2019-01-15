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
import ecdsa
from base58 import b58encode_check
from utils import Utils
from wallet import Wallet

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def with_lock(lock):
    def dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return dec


from ds.Block import Block
from ds.OutPoint import OutPoint
from ds.TxIn import TxIn
from ds.TxOut import TxOut
from ds.UnspentTxOut import UnspentTxOut
from ds.Transaction import Transaction


from ds.MerkleNode import MerkleNode
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from params.Params import Params


from _thread import RLock

# Chain
# ----------------------------------------------------------------------------

# The highest proof-of-work, valid blockchain.
#
# #realname chainActive
active_chain: Iterable[Block] = [Params.genesis_block]

# Branches off of the main chain.
side_branches: Iterable[Iterable[Block]] = []

# Synchronize access to the active chain and side branches.
chain_lock = threading.RLock()





orphan_blocks: Iterable[Block] = []

# Used to signify the active chain in `locate_block`.
ACTIVE_CHAIN_IDX = 0


@with_lock(chain_lock)
def get_current_height(): return len(active_chain)


@with_lock(chain_lock)
def txn_iterator(chain):
    return (
        (txn, block, height)
        for height, block in enumerate(chain) for txn in block.txns)


@with_lock(chain_lock)
def locate_block(block_hash: str, chain=None) -> (Block, int, int):
    chains = [chain] if chain else [active_chain, *side_branches]

    for chain_idx, chain in enumerate(chains):
        for height, block in enumerate(chain):
            if block.id == block_hash:
                return (block, height, chain_idx)
    return (None, None, None)


@with_lock(chain_lock)
def connect_block(block: Union[str, Block], peers: Iterable[Peer], mempool: MemPool, utxo_set:UTXO_Set,
                  doing_reorg=False,
                  ) -> Union[None, Block]:
    """Accept a block and return the chain index we append it to."""
    # Only exit early on already seen in active_chain when reorging.
    search_chain = active_chain if doing_reorg else None

    if locate_block(block.id, chain=search_chain)[0]:
        logger.debug(f'ignore block already seen: {block.id}')
        return None

    try:
        block, chain_idx = validate_block(block)
    except BlockValidationError as e:
        logger.exception('block %s failed validation', block.id)
        if e.to_orphan:
            logger.info(f"saw orphan block {block.id}")
            orphan_blocks.append(e.to_orphan)
        return None

    # If `validate_block()` returned a non-existent chain index, we're
    # creating a new side branch.
    if chain_idx != ACTIVE_CHAIN_IDX and len(side_branches) < chain_idx:
        logger.info(
            f'creating a new side branch (idx {chain_idx}) '
            f'for block {block.id}')
        side_branches.append([])

    logger.info(f'connecting block {block.id} to chain {chain_idx}')
    chain = (active_chain if chain_idx == ACTIVE_CHAIN_IDX else
             side_branches[chain_idx - 1])
    chain.append(block)

    # If we added to the active chain, perform upkeep on utxo_set and mempool.
    if chain_idx == ACTIVE_CHAIN_IDX:
        for tx in block.txns:
            mempool.mempool.pop(tx.id, None)

            if not tx.is_coinbase:
                for txin in tx.txins:
                    utxo_set.rm_from_utxo(*txin.to_spend)
            for i, txout in enumerate(tx.txouts):
                utxo_set.add_to_utxo(txout, tx, i, tx.is_coinbase, len(chain))

    if (not doing_reorg and reorg_if_necessary()) or \
            chain_idx == ACTIVE_CHAIN_IDX:
        mine_interrupt.set()
        logger.info(
            f'block accepted '
            f'height={len(active_chain) - 1} txns={len(block.txns)}')

    for peer in peers:
        Utils.send_to_peer(block, peer)

    return chain_idx


@with_lock(chain_lock)
def disconnect_block(block, mempool:MemPool, utxo_set: UTXO_Set, chain=None):
    chain = chain or active_chain
    assert block == chain[-1], "Block being disconnected must be tip."

    for tx in block.txns:
        mempool.mempool[tx.id] = tx

        # Restore UTXO set to what it was before this block.
        for txin in tx.txins:
            if txin.to_spend:  # Account for degenerate coinbase txins.
                utxo_set.add_to_utxo(*find_txout_for_txin(txin, chain))
        for i in range(len(tx.txouts)):
            utxo_set.rm_from_utxo(tx.id, i)

    logger.info(f'block {block.id} disconnected')
    return chain.pop()


def find_txout_for_txin(txin, chain):
    txid, txout_idx = txin.to_spend

    for tx, block, height in txn_iterator(chain):
        if tx.id == txid:
            txout = tx.txouts[txout_idx]
            return (txout, tx, txout_idx, tx.is_coinbase, height)


@with_lock(chain_lock)
def reorg_if_necessary() -> bool:
    reorged = False
    frozen_side_branches = list(side_branches)  # May change during this call.

    # TODO should probably be using `chainwork` for the basis of
    # comparison here.
    for branch_idx, chain in enumerate(frozen_side_branches, 1):
        fork_block, fork_idx, _ = locate_block(
            chain[0].prev_block_hash, active_chain)
        active_height = len(active_chain)
        branch_height = len(chain) + fork_idx

        if branch_height > active_height:
            logger.info(
                f'attempting reorg of idx {branch_idx} to active_chain: '
                f'new height of {branch_height} (vs. {active_height})')
            reorged |= try_reorg(chain, branch_idx, fork_idx)

    return reorged


@with_lock(chain_lock)
def try_reorg(branch, branch_idx, fork_idx) -> bool:
    # Use the global keyword so that we can actually swap out the reference
    # in case of a reorg.
    global active_chain
    global side_branches

    fork_block = active_chain[fork_idx]

    def disconnect_to_fork():
        while active_chain[-1].id != fork_block.id:
            yield disconnect_block(active_chain[-1])

    old_active = list(disconnect_to_fork())[::-1]

    assert branch[0].prev_block_hash == active_chain[-1].id

    def rollback_reorg():
        logger.info(f'reorg of idx {branch_idx} to active_chain failed')
        list(disconnect_to_fork())  # Force the gneerator to eval.

        for block in old_active:
            assert connect_block(block, doing_reorg=True) == ACTIVE_CHAIN_IDX

    for block in branch:
        connected_idx = connect_block(block, doing_reorg=True)
        if connected_idx != ACTIVE_CHAIN_IDX:
            rollback_reorg()
            return False

    # Fix up side branches: remove new active, add old active.
    side_branches.pop(branch_idx - 1)
    side_branches.append(old_active)

    logger.info(
        'chain reorg! New height: %s, tip: %s',
        len(active_chain), active_chain[-1].id)

    return True





# Chain Persistance
# ----------------------------------------------------------------------------

CHAIN_PATH = os.environ.get('TC_CHAIN_PATH', 'chain.dat')

@with_lock(chain_lock)
def save_to_disk():
    with open(CHAIN_PATH, "wb") as f:
        logger.info(f"saving chain with {len(active_chain)} blocks")
        f.write(Utils.encode_socket_data(list(active_chain)))

@with_lock(chain_lock)
def load_from_disk():
    if not os.path.isfile(CHAIN_PATH):
        return
    try:
        with open(CHAIN_PATH, "rb") as f:
            msg_len = int(binascii.hexlify(f.read(4) or b'\x00'), 16)
            new_blocks = Utils.deserialize(f.read(msg_len))
            logger.info(f"loading chain from disk with {len(new_blocks)} blocks")
            for block in new_blocks:
                connect_block(block)
    except Exception:
        logger.exception('load chain failed, starting from genesis')



# Proof of work
# ----------------------------------------------------------------------------

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


def assemble_and_solve_block(pay_coinbase_to_addr, mempool:MemPool, wallet, utxo_set:UTXO_Set, txns=None):
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
        block = mempool.select_from_mempool(block)

    fees = block.calculate_fees(utxo_set.get())
    my_address = wallet.get()[2]
    coinbase_txn = Transaction.create_coinbase(
        my_address, (Block.get_block_subsidy(active_chain) + fees), len(active_chain))
    block = block._replace(txns=[coinbase_txn, *block.txns])
    block = block._replace(merkle_hash=MerkleNode.get_merkle_root_of_txns(block.txns).val)

    if len(Utils.serialize(block)) > Params.MAX_BLOCK_SERIALIZED_SIZE:
        raise ValueError('txns specified create a block too large')

    return mine(block)



# Signal to communicate to the mining thread that it should stop mining because
# we've updated the chain with a new block.
mine_interrupt = threading.Event()


def mine(block):
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


# Validation
# ----------------------------------------------------------------------------







# Signal when the initial block download has completed.
ibd_done = threading.Event()


WALLET_PATH = 'wallet.dat'
PORT = 9999




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
