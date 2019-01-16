from ds.Block import Block
from ds.TxIn import TxIn

from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)

class BlockChain(object):

    def __init__(self, idx=0, chain=[]):
        self.idx = idx
        self.chain: Iterable[Block] = chain

    @property
    def height(self):
        return len(self.chain)

    @property
    def idx(self):
        return self.idx

    def find_txout_for_txin(self, txin: TxIn):

        def _txn_iterator(chain: Iterable[Block]):
            return (
                (txn, block, height)
                for height, block in enumerate(chain) for txn in block.txns)

        txid, txout_idx = txin.to_spend

        for tx, block, height in _txn_iterator(self.chain):
            if tx.id == txid:
                txout = tx.txouts[txout_idx]
                return (txout, tx, txout_idx, tx.is_coinbase, height)



