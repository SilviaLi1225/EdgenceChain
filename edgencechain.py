#!/usr/bin/env python3
"""
⛼  edgencechain

  putting the rough in "rough consensus"


Some terminology:

- Chain: an ordered list of Blocks, each of which refers to the last and
      cryptographically preserves a history of Transactions.

- Transaction (or tx or txn): a list of inputs (i.e. past outputs being spent)
    and outputs which declare value assigned to the hash of a public key.

- PoW (proof of work): the solution to a puzzle which allows the acceptance
    of an additional Block onto the chain.

- Reorg: chain reorganization. When a side branch overtakes the main chain.


An incomplete list of unrealistic simplifications:

- Byte encoding and endianness are very important when serializing a
  data structure to be hashed in Bitcoin and are not reproduced
  faithfully here. In fact, serialization of any kind here is slipshod and
  in many cases relies on implicit expectations about Python JSON
  serialization.

- Transaction types are limited to P2PKH.

- Initial Block Download eschews `getdata` and instead returns block payloads
  directly in `inv`.

- Peer "discovery" is done through environment variable hardcoding. In
  bitcoin core, this is done with DNS seeds.
  See https://bitcoin.stackexchange.com/a/3537/56368


Resources:

- https://en.bitcoin.it/wiki/Protocol_rules
- https://en.bitcoin.it/wiki/Protocol_documentation
- https://bitcoin.org/en/developer-guide
- https://github.com/bitcoinbook/bitcoinbook/blob/second_edition/ch06.asciidoc


TODO:

- deal with orphan blocks
- keep the mempool heap sorted by fee
- make use of Transaction.locktime
? make use of TxIn.sequence; i.e. replace-by-fee

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

import ecdsa
from base58 import b58encode_check


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)







# Chain
# ----------------------------------------------------------------------------

genesis_block = Block(
    version=0, prev_block_hash=None,
    merkle_hash=(
        '7118894203235a955a908c0abfc6d8fe6edec47b0a04ce1bf7263da3b4366d22'),
    timestamp=1501821412, bits=24, nonce=10126761,
    txns=[Transaction(
        txins=[TxIn(
            to_spend=None, unlock_sig=b'0', unlock_pk=None, sequence=0)],
        txouts=[TxOut(
            value=5000000000,
            to_address='143UVyz7ooiAv1pMqbwPPpnH4BV9ifJGFF')], locktime=None)])

# The highest proof-of-work, valid blockchain.
#
# #realname chainActive
active_chain: Iterable[Block] = [genesis_block]

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
def connect_block(block: Union[str, Block],
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
            mempool.pop(tx.id, None)

            if not tx.is_coinbase:
                for txin in tx.txins:
                    rm_from_utxo(*txin.to_spend)
            for i, txout in enumerate(tx.txouts):
                add_to_utxo(txout, tx, i, tx.is_coinbase, len(chain))

    if (not doing_reorg and reorg_if_necessary()) or \
            chain_idx == ACTIVE_CHAIN_IDX:
        mine_interrupt.set()
        logger.info(
            f'block accepted '
            f'height={len(active_chain) - 1} txns={len(block.txns)}')

    for peer in peer_hostnames:
        send_to_peer(block, peer)

    return chain_idx


@with_lock(chain_lock)
def disconnect_block(block, chain=None):
    chain = chain or active_chain
    assert block == chain[-1], "Block being disconnected must be tip."

    for tx in block.txns:
        mempool[tx.id] = tx

        # Restore UTXO set to what it was before this block.
        for txin in tx.txins:
            if txin.to_spend:  # Account for degenerate coinbase txins.
                add_to_utxo(*find_txout_for_txin(txin, chain))
        for i in range(len(tx.txouts)):
            rm_from_utxo(tx.id, i)

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


def get_median_time_past(num_last_blocks: int) -> int:
    """Grep for: GetMedianTimePast."""
    last_n_blocks = active_chain[::-1][:num_last_blocks]

    if not last_n_blocks:
        return 0

    return last_n_blocks[len(last_n_blocks) // 2].timestamp


# Chain Persistance
# ----------------------------------------------------------------------------

CHAIN_PATH = os.environ.get('TC_CHAIN_PATH', 'chain.dat')

@with_lock(chain_lock)
def save_to_disk():
    with open(CHAIN_PATH, "wb") as f:
        logger.info(f"saving chain with {len(active_chain)} blocks")
        f.write(encode_socket_data(list(active_chain)))

@with_lock(chain_lock)
def load_from_disk():
    if not os.path.isfile(CHAIN_PATH):
        return
    try:
        with open(CHAIN_PATH, "rb") as f:
            msg_len = int(binascii.hexlify(f.read(4) or b'\x00'), 16)
            new_blocks = deserialize(f.read(msg_len))
            logger.info(f"loading chain from disk with {len(new_blocks)} blocks")
            for block in new_blocks:
                connect_block(block)
    except Exception:
        logger.exception('load chain failed, starting from genesis')


# UTXO set
# ----------------------------------------------------------------------------

utxo_set: Mapping[OutPoint, UnspentTxOut] = {}


def add_to_utxo(txout, tx, idx, is_coinbase, height):
    utxo = UnspentTxOut(
        *txout,
        txid=tx.id, txout_idx=idx, is_coinbase=is_coinbase, height=height)

    logger.info(f'adding tx outpoint {utxo.outpoint} to utxo_set')
    utxo_set[utxo.outpoint] = utxo


def rm_from_utxo(txid, txout_idx):
    del utxo_set[OutPoint(txid, txout_idx)]


def find_utxo_in_list(txin, txns) -> UnspentTxOut:
    txid, txout_idx = txin.to_spend
    try:
        txout = [t for t in txns if t.id == txid][0].txouts[txout_idx]
    except Exception:
        return None

    return UnspentTxOut(
        *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=txout_idx)


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


def assemble_and_solve_block(pay_coinbase_to_addr, txns=None):
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
        block = select_from_mempool(block)

    fees = calculate_fees(block)
    my_address = init_wallet()[2]
    coinbase_txn = Transaction.create_coinbase(
        my_address, (get_block_subsidy() + fees), len(active_chain))
    block = block._replace(txns=[coinbase_txn, *block.txns])
    block = block._replace(merkle_hash=get_merkle_root_of_txns(block.txns).val)

    if len(serialize(block)) > Params.MAX_BLOCK_SERIALIZED_SIZE:
        raise ValueError('txns specified create a block too large')

    return mine(block)


def calculate_fees(block) -> int:
    """
    Given the txns in a Block, subtract the amount of coin output from the
    inputs. This is kept as a reward by the miner.
    """
    fee = 0

    def utxo_from_block(txin):
        tx = [t.txouts for t in block.txns if t.id == txin.to_spend.txid]
        return tx[0][txin.to_spend.txout_idx] if tx else None

    def find_utxo(txin):
        return utxo_set.get(txin.to_spend) or utxo_from_block(txin)

    for txn in block.txns:
        spent = sum(find_utxo(i).value for i in txn.txins)
        sent = sum(o.value for o in txn.txouts)
        fee += (spent - sent)

    return fee


def get_block_subsidy() -> int:
    halvings = len(active_chain) // Params.HALVE_SUBSIDY_AFTER_BLOCKS_NUM

    if halvings >= 64:
        return 0

    return 50 * Params.LET_PER_COIN // (2 ** halvings)


# Signal to communicate to the mining thread that it should stop mining because
# we've updated the chain with a new block.
mine_interrupt = threading.Event()


def mine(block):
    start = time.time()
    nonce = 0
    target = (1 << (256 - block.bits))
    mine_interrupt.clear()

    while int(sha256d(block.header(nonce)), 16) >= target:
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
        my_address = init_wallet()[2]
        block = assemble_and_solve_block(my_address)

        if block:
            connect_block(block)
            save_to_disk()


# Validation
# ----------------------------------------------------------------------------


def validate_txn(txn: Transaction,
                 as_coinbase: bool = False,
                 siblings_in_block: Iterable[Transaction] = None,
                 allow_utxo_from_mempool: bool = True,
                 ) -> Transaction:
    """
    Validate a single transaction. Used in various contexts, so the
    parameters facilitate different uses.
    """
    txn.validate_basics(as_coinbase=as_coinbase)

    available_to_spend = 0

    for i, txin in enumerate(txn.txins):
        utxo = utxo_set.get(txin.to_spend)

        if siblings_in_block:
            utxo = utxo or find_utxo_in_list(txin, siblings_in_block)

        if allow_utxo_from_mempool:
            utxo = utxo or find_utxo_in_mempool(txin)

        if not utxo:
            raise TxnValidationError(
                f'Could find no UTXO for TxIn[{i}] -- orphaning txn',
                to_orphan=txn)

        if utxo.is_coinbase and \
                (get_current_height() - utxo.height) < \
                Params.COINBASE_MATURITY:
            raise TxnValidationError(f'Coinbase UTXO not ready for spend')

        try:
            validate_signature_for_spend(txin, utxo, txn)
        except TxUnlockError:
            raise TxnValidationError(f'{txin} is not a valid spend of {utxo}')

        available_to_spend += utxo.value

    if available_to_spend < sum(o.value for o in txn.txouts):
        raise TxnValidationError('Spend value is more than available')

    return txn


def validate_signature_for_spend(txin, utxo: UnspentTxOut, txn):
    pubkey_as_addr = pubkey_to_address(txin.unlock_pk)
    verifying_key = ecdsa.VerifyingKey.from_string(
        txin.unlock_pk, curve=ecdsa.SECP256k1)

    if pubkey_as_addr != utxo.to_address:
        raise TxUnlockError("Pubkey doesn't match")

    try:
        spend_msg = build_spend_message(
            txin.to_spend, txin.unlock_pk, txin.sequence, txn.txouts)
        verifying_key.verify(txin.unlock_sig, spend_msg)
    except Exception:
        logger.exception('Key verification failed')
        raise TxUnlockError("Signature doesn't match")

    return True


def build_spend_message(to_spend, pk, sequence, txouts) -> bytes:
    """This should be ~roughly~ equivalent to SIGHASH_ALL."""
    return sha256d(
        serialize(to_spend) + str(sequence) +
        binascii.hexlify(pk).decode() + serialize(txouts)).encode()


@with_lock(chain_lock)
def validate_block(block: Block) -> Block:
    if not block.txns:
        raise BlockValidationError('txns empty')

    if block.timestamp - time.time() > Params.MAX_FUTURE_BLOCK_TIME:
        raise BlockValidationError('Block timestamp too far in future')

    if int(block.id, 16) > (1 << (256 - block.bits)):
        raise BlockValidationError("Block header doesn't satisfy bits")

    if [i for (i, tx) in enumerate(block.txns) if tx.is_coinbase] != [0]:
        raise BlockValidationError('First txn must be coinbase and no more')

    try:
        for i, txn in enumerate(block.txns):
            txn.validate_basics(as_coinbase=(i == 0))
    except TxnValidationError:
        logger.exception(f"Transaction {txn} in {block} failed to validate")
        raise BlockValidationError('Invalid txn {txn.id}')

    if get_merkle_root_of_txns(block.txns).val != block.merkle_hash:
        raise BlockValidationError('Merkle hash invalid')

    if block.timestamp <= get_median_time_past(11):
        raise BlockValidationError('timestamp too old')

    if not block.prev_block_hash and not active_chain:
        # This is the genesis block.
        prev_block_chain_idx = ACTIVE_CHAIN_IDX
    else:
        prev_block, prev_block_height, prev_block_chain_idx = locate_block(
            block.prev_block_hash)

        if not prev_block:
            raise BlockValidationError(
                f'prev block {block.prev_block_hash} not found in any chain',
                to_orphan=block)

        # No more validation for a block getting attached to a branch.
        if prev_block_chain_idx != ACTIVE_CHAIN_IDX:
            return block, prev_block_chain_idx

        # Prev. block found in active chain, but isn't tip => new fork.
        elif prev_block != active_chain[-1]:
            return block, prev_block_chain_idx + 1  # Non-existent

    if get_next_work_required(block.prev_block_hash) != block.bits:
        raise BlockValidationError('bits is incorrect')

    for txn in block.txns[1:]:
        try:
            validate_txn(txn, siblings_in_block=block.txns[1:],
                         allow_utxo_from_mempool=False)
        except TxnValidationError:
            msg = f"{txn} failed to validate"
            logger.exception(msg)
            raise BlockValidationError(msg)

    return block, prev_block_chain_idx


# mempool
# ----------------------------------------------------------------------------

# Set of yet-unmined transactions.
mempool: Dict[str, Transaction] = {}

# Set of orphaned (i.e. has inputs referencing yet non-existent UTXOs)
# transactions.
orphan_txns: Iterable[Transaction] = []


def find_utxo_in_mempool(txin) -> UnspentTxOut:
    txid, idx = txin.to_spend

    try:
        txout = mempool[txid].txouts[idx]
    except Exception:
        logger.debug("Couldn't find utxo in mempool for %s", txin)
        return None

    return UnspentTxOut(
        *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=idx)


def select_from_mempool(block: Block) -> Block:
    """Fill a Block with transactions from the mempool."""
    added_to_block = set()

    def check_block_size(b) -> bool:
        return len(serialize(block)) < Params.MAX_BLOCK_SERIALIZED_SIZE

    def try_add_to_block(block, txid) -> Block:
        if txid in added_to_block:
            return block

        tx = mempool[txid]

        # For any txin that can't be found in the main chain, find its
        # transaction in the mempool (if it exists) and add it to the block.
        for txin in tx.txins:
            if txin.to_spend in utxo_set:
                continue

            in_mempool = find_utxo_in_mempool(txin)

            if not in_mempool:
                logger.debug(f"Couldn't find UTXO for {txin}")
                return None

            block = try_add_to_block(block, in_mempool.txid)
            if not block:
                logger.debug(f"Couldn't add parent")
                return None

        newblock = block._replace(txns=[*block.txns, tx])

        if check_block_size(newblock):
            logger.debug(f'added tx {tx.id} to block')
            added_to_block.add(txid)
            return newblock
        else:
            return block

    for txid in mempool:
        newblock = try_add_to_block(block, txid)

        if check_block_size(newblock):
            block = newblock
        else:
            break

    return block


def add_txn_to_mempool(txn: Transaction):
    if txn.id in mempool:
        logger.info(f'txn {txn.id} already seen')
        return

    try:
        txn = validate_txn(txn)
    except TxnValidationError as e:
        if e.to_orphan:
            logger.info(f'txn {e.to_orphan.id} submitted as orphan')
            orphan_txns.append(e.to_orphan)
        else:
            logger.exception(f'txn rejected')
    else:
        logger.info(f'txn {txn.id} added to mempool')
        mempool[txn.id] = txn

        for peer in peer_hostnames:
            send_to_peer(txn, peer)




# Peer-to-peer
# ----------------------------------------------------------------------------

peer_hostnames = {p for p in os.environ.get('TC_PEERS', '').split(',') if p}

# Signal when the initial block download has completed.
ibd_done = threading.Event()




# Wallet
# ----------------------------------------------------------------------------

WALLET_PATH = os.environ.get('TC_WALLET_PATH', 'wallet.dat')





# Misc. utilities
# ----------------------------------------------------------------------------






# Main
# ----------------------------------------------------------------------------

PORT = os.environ.get('TC_PORT', 9999)


def main():
    load_from_disk()

    workers = []
    server = ThreadedTCPServer(('0.0.0.0', PORT), TCPHandler)

    def start_worker(fnc):
        workers.append(threading.Thread(target=fnc, daemon=True))
        workers[-1].start()

    logger.info(f'[p2p] listening on {PORT}')
    start_worker(server.serve_forever)

    if peer_hostnames:
        logger.info(
            f'start initial block download from {len(peer_hostnames)} peers')
        send_to_peer(GetBlocksMsg(active_chain[-1].id))
        ibd_done.wait(60.)  # Wait a maximum of 60 seconds for IBD to complete.

    start_worker(mine_forever)
    [w.join() for w in workers]


if __name__ == '__main__':
    signing_key, verifying_key, my_address = init_wallet()
    main()
