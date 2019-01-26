import logging
import os
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)

from ds.UnspentTxOut import UnspentTxOut
from ds.OutPoint import OutPoint
from ds.BaseUTXO_Set import BaseUTXO_Set



logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



  # UTXO set


class UTXO_Set(BaseUTXO_Set):

    def __init__(self):
        self.utxoSet: Mapping[OutPoint, UnspentTxOut] = {}
    def get(self):
        return self.utxoSet


    def add_to_utxo(self, txout, tx, idx, is_coinbase, height):
        utxo = UnspentTxOut(
            *txout,
            txid=tx.id, txout_idx=idx, is_coinbase=is_coinbase, height=height)

        logger.info(f'adding tx outpoint {utxo.outpoint} to utxo_set')
        self.utxoSet[utxo.outpoint] = utxo


    def rm_from_utxo(self, txid, txout_idx):
        del self.utxoSet[OutPoint(txid, txout_idx)]

    @classmethod
    def find_utxo_in_list(cls, txin, txns) -> UnspentTxOut:
        txid, txout_idx = txin.to_spend
        try:
            txout = [t for t in txns if t.id == txid][0].txouts[txout_idx]
        except Exception:
            return None

        return UnspentTxOut(
            *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=txout_idx)



