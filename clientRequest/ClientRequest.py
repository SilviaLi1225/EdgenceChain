from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
import os,logging,binascii


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from utils.Utils import Utils


class GetUTXOsMsg(NamedTuple):  # List all UTXOs
    def handle(self, sock, peer):
        sock.sendall(Utils.encode_socket_data(list(utxo_set.items())))


class GetMempoolMsg(NamedTuple):  # List the mempool
    def handle(self, sock, peer):
        sock.sendall(Utils.encode_socket_data(list(mempool.keys())))


class GetActiveChainMsg(NamedTuple):  # Get the active chain in its entirety.
    def handle(self, sock, peer):
        sock.sendall(Utils.encode_socket_data(list(active_chain)))

