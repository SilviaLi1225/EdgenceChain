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


from ds.UnspentTxOut import UnspentTxOut
from ds.OutPoint import OutPoint
from ds.Transaction import Transaction
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from utils import Utils
from params.Params import Params
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain
from ds.Block import Block

from persistence import Persistence
from wallet.Wallet import Wallet
from p2p.Peer import Peer
import ecdsa
from base58 import b58encode_check


from _thread import RLock


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class EdgenceChain(object):

    def __init__(self):

        self.active_chain: BlockChain = BlockChain(idx=Params.ACTIVE_CHAIN_IDX, chain=[Params.genesis_block])
        self.side_branches: Iterable[BlockChain] = []
        self.orphan_blocks: Iterable[Block] = []
        self.utxo_set: UTXO_Set = UTXO_Set()
        self.mempool: MemPool = MemPool()
        self.wallet = Wallet.init_wallet(Params.WALLET_FILE)
        self.peers = Peer.init_peers(Params.PEERS_FILE)

        Persistence.load_from_disk(self.active_chain, self.utxo_set, Params.CHAIN_FILE)






