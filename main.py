#!/usr/bin/env python3
"""
⛼  edgencechain

  putting the rough in "rough consensus"


"""
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

from p2p.P2P import (GetBlocksMsg, InvMsg, ThreadedTCPServer, TCPHandler)
from p2p.Peer import Peer
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain

import ecdsa
from base58 import b58encode_check
from utils import Utils
from wallet import Wallet

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)






from ds.Block import Block
from ds.OutPoint import OutPoint
from ds.TxIn import TxIn
from ds.TxOut import TxOut
from ds.UnspentTxOut import UnspentTxOut
from ds.Transaction import Transaction


from ds.MerkleNode import MerkleNode
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from params.Params import Params


from utils.Utils import Utils
from consensus.Consensus import PoW

#
# #realname chainActive











def main():
    load_from_disk()
    wallet = Wallet.init_wallet(WALLET_PATH)
    peers = Peer.init_peers()
    utxo_set = UTXO_Set()   #utxo_set.get()
    mempool = MemPool()

    workers = []
    server = ThreadedTCPServer(('0.0.0.0', PORT), TCPHandler)

    def start_worker(fnc):
        workers.append(threading.Thread(target=fnc, daemon=True))
        workers[-1].start()

    logger.info(f'[p2p] listening on {PORT}')
    start_worker(server.serve_forever)

    if peers:
        logger.info(
            f'start initial block download from {len(peers)} peers')
        peer = random.choice(list(peers))
        Utils.send_to_peer(GetBlocksMsg(active_chain[-1].id), peer)
        ibd_done.wait(60.)  # Wait a maximum of 60 seconds for IBD to complete.

    start_worker(mine_forever)
    [w.join() for w in workers]


if __name__ == '__main__':

    main()
