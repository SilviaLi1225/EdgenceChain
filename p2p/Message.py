from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
from ds.Block import Block
from ds.Transaction import Transaction
from ds.UnspentTxOut import UnspentTxOut
from p2p.Peer import Peer

from enum import Enum, unique



class Actions:
    BlocksSyncReq = 0
    BlocksSyncGet = 1
    TxStatusReq  = 2
    UTXO4Addr    = 3
    Balance4Addr = 4
    TxRev        = 5
    BlockRev     = 6
    TxStatusRev = 7
    UTXO4AddrRev = 8
    Balance4AddrRev = 9


class Message(NamedTuple):
    action: int
    data: Union[str, Iterable[Block], Iterable[UnspentTxOut], int, Transaction, Block, Iterable[Peer]]
    port: Union[None, int] = None





