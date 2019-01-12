# mempool  Set of yet-unmined transactions.
import logging
import os
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)

from ds.Block  import Block
from ds.UnspentTxOut import UnspentTxOut
from ds.Transaction import (OutPoint, TxIn, TxOut, Transaction)
from ds.UTXO_Set import UTXO_Set
from params.Params import Params
from utils.Utils import Utils
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class MemPool(object):
    def __init__(self):
        self.mempool: Dict[str, Transaction] = {}

        # Set of orphaned (i.e. has inputs referencing yet non-existent UTXOs)
        # transactions.
        self.orphan_txns: Iterable[Transaction] = []
    def get(self):
        return self.mempool


    def find_utxo_in_mempool(self,txin) -> UnspentTxOut:
        txid, idx = txin.to_spend

        try:
            txout = self.mempool[txid].txouts[idx]
        except Exception:
            logger.debug("Couldn't find utxo in mempool for %s", txin)
            return None

        return UnspentTxOut(
            *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=idx)


    def select_from_mempool(self, block: Block, utxo_set:UTXO_Set) -> Block:
        """Fill a Block with transactions from the mempool."""
        added_to_block = set()

        def check_block_size(block) -> bool:
            return len(Utils.serialize(block)) < Params.MAX_BLOCK_SERIALIZED_SIZE

        def try_add_to_block(block, txid) -> Block:
            if txid in added_to_block:
                return block

            tx = self.mempool[txid]

            # For any txin that can't be found in the main chain, find its
            # transaction in the mempool (if it exists) and add it to the block.

            for txin in tx.txins:
                if txin.to_spend in utxo_set.get():
                    continue

                in_mempool = self.find_utxo_in_mempool(txin)

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

        for txid in self.mempool:
            newblock = try_add_to_block(block, txid)

            if check_block_size(newblock):
                block = newblock
            else:
                break

        return block


    def add_txn_to_mempool(self, txn: Transaction, peers):
        if txn.id in self.mempool:
            logger.info(f'txn {txn.id} already seen')
            return

        try:
            txn.validate_txn()
        except TxnValidationError as e:
            if e.to_orphan:
                logger.info(f'txn {e.to_orphan.id} submitted as orphan')
                self.orphan_txns.append(e.to_orphan)
            else:
                logger.exception(f'txn rejected')
        else:
            logger.info(f'txn {txn.id} added to mempool')
            self.mempool[txn.id] = txn

            for peer in peers:
                Utils.send_to_peer(txn, peer)
