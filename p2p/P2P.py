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

from dataStructure.Block  import (OutPoint, TxIn, TxOut, UnspentTxOut, Transaction,
                                  Block)
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from utils import Utils
from params.Params import Params

import ecdsa
from base58 import b58encode_check


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class GetBlocksMsg(NamedTuple):  # Request blocks during initial sync
    """
    See https://bitcoin.org/en/developer-guide#blocks-first
    """
    from_blockid: str

    CHUNK_SIZE = 50

    def handle(self, sock, peer_hostname):
        logger.debug(f"[p2p] recv getblocks from {peer_hostname}")

        _, height, _ = locate_block(self.from_blockid, active_chain)

        # If we don't recognize the requested hash as part of the active
        # chain, start at the genesis block.
        height = height or 1

        with chain_lock:
            blocks = active_chain[height:(height + self.CHUNK_SIZE)]

        logger.debug(f"[p2p] sending {len(blocks)} to {peer_hostname}")
        Utils.send_to_peer(InvMsg(blocks), peer_hostname)


class InvMsg(NamedTuple):  # Convey blocks to a peer who is doing initial sync
    blocks: Iterable[str]

    def handle(self, sock, peer_hostname):
        logger.info(f"[p2p] recv inv from {peer_hostname}")

        new_blocks = [b for b in self.blocks if not locate_block(b.id)[0]]

        if not new_blocks:
            logger.info('[p2p] initial block download complete')
            ibd_done.set()
            return

        for block in new_blocks:
            connect_block(block)

        new_tip_id = active_chain[-1].id
        logger.info(f'[p2p] continuing initial block download at {new_tip_id}')

        with chain_lock:
            # "Recursive" call to continue the initial block sync.
            Utils.send_to_peer(GetBlocksMsg(new_tip_id))





class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class TCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = Utils.read_all_from_socket(self.request)
        peer_hostname = self.request.getpeername()[0]
        peer_hostnames.add(peer_hostname)

        if hasattr(data, 'handle') and isinstance(data.handle, Callable):
            logger.info(f'received msg {data} from peer {peer_hostname}')
            data.handle(self.request, peer_hostname)
        elif isinstance(data, Transaction):
            logger.info(f"received txn {data.id} from peer {peer_hostname}")
            add_txn_to_mempool(data)
        elif isinstance(data, Block):
            logger.info(f"received block {data.id} from peer {peer_hostname}")
            connect_block(data)
