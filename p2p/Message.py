from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
from ds.Block import Block
from ds.Transaction import Transaction
from ds.UnspentTxOut import UnspentTxOut

from enum import Enum, unique

@unique
class Action(Enum):
    BlockSyncReq = 0
    BlockSyncGet = 1
    TxStatusReq  = 2
    UTXO4Addr    = 3
    Balance4Addr = 4
    TxRev        = 5
    BlockRev     = 6

class Message(NamedTuple):
    action: int
    data: Union[str, Iterable[Block], str, Iterable[UnspentTxOut], int, Transaction, Block]


