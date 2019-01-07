import logging
import os
from typing import (Callable, Dict, Iterable, Mapping, NamedTuple, Tuple,
                    Union, get_type_hints)
from edgencechain.definitions import (Block, BlockValidationError, OutPoint,
                                      Transaction, TxIn, TxnValidationError,
                                      TxOut, TxUnlockError, UnspentTxOut)
from edgencechain.helpers import serialize
from edgencechain.params import Params
from edgencechain.chain import validate_txn, peer_hostnames, send_to_peer, utxo_set

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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
