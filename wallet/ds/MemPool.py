# mempool  Set of yet-unmined transactions.
import logging
import os
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)

from ds.Block import Block
from ds.UnspentTxOut import UnspentTxOut
from ds.Transaction import Transaction
from ds.TxIn import TxIn
from ds.UTXO_Set import UTXO_Set
from params.Params import Params
from utils.Utils import Utils
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from p2p.Peer import Peer
from ds.BaseMemPool import BaseMemPool
from ds.BaseUTXO_Set import BaseUTXO_Set

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class MemPool(BaseMemPool):
    def __init__(self):
        self.mempool: Dict[str, Transaction] = {}
        self.orphan_txns: Iterable[Transaction] = []

    def get(self):
        return self.mempool

    def find_utxo_in_mempool(self, txin: TxIn) -> UnspentTxOut:
        txid, idx = txin.to_spend

        try:
            txout = self.mempool[txid].txouts[idx]
        except Exception:
            logger.debug(f"[ds] could not find utxo in mempool for txin {txin}")
            return None

        return UnspentTxOut(
            *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=idx)


    def select_from_mempool(self, block: Block, utxo_set: UTXO_Set) -> Block:
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
                    logger.debug(f"[ds] Couldn't find UTXO for {txin}")
                    return None

                block = try_add_to_block(block, in_mempool.txid)
                if not block:
                    logger.debug(f"[ds] Couldn't add parent")
                    return None

            newblock = block._replace(txns=[*block.txns, tx])

            if check_block_size(newblock):
                logger.debug(f'[ds] added tx {tx.id} to block')
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


    def add_txn_to_mempool(self, txn: Transaction, utxo_set: BaseUTXO_Set) -> bool:
        if txn.id in self.mempool:
            logger.info(f'[ds] txn {txn.id} already seen')
            return None

        try:
            txn.validate_txn(utxo_set, self.mempool)
        except TxnValidationError as e:
            if e.to_orphan:
                logger.info(f'[ds] txn {e.to_orphan.id} submitted as orphan')
                self.orphan_txns.append(e.to_orphan)
            else:
                logger.exception(f'[ds] txn rejected')
            return None
        else:
            logger.info(f'[ds] txn {txn.id} added to mempool')
            self.mempool[txn.id] = txn

            return True

