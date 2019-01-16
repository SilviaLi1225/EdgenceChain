from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
import os,logging,binascii


from ds.MemPool import MemPool
from ds.UTXO_Set import UTXO_Set

from utils.Utils import Utils


class GetUTXOsMsg(NamedTuple):  # List all UTXOs
    def handle(self, sock, utxo_set:UTXO_Set):
        sock.sendall(Utils.encode_socket_data(list(utxo_set.get().items())))

class GetMempoolMsg(NamedTuple):  # List the mempool
    def handle(self, sock, mempool:MemPool):
        sock.sendall(Utils.encode_socket_data(list(mempool.get().keys())))



