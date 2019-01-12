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
from  ds.Transaction import (OutPoint, TxIn, TxOut, Transaction)
from ds.UTXO_Set import UTXO_Set
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from utils import Utils
from params.Params import Params

import ecdsa
from base58 import b58encode_check


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)










class Block(NamedTuple):
    # A version integer.
    version: int

    # A hash of the previous block's header.
    prev_block_hash: str

    # A hash of the Merkle tree containing all txns.
    merkle_hash: str

    # A UNIX timestamp of when this block was created.
    timestamp: int

    # The difficulty target; i.e. the hash of this block header must be under
    # (2 ** 256 >> bits) to consider work proved.
    bits: int

    # The value that's incremented in an attempt to get the block header to
    # hash to a value below `bits`.
    nonce: int

    txns: Iterable[Transaction]

    def header(self, nonce=None) -> str:
        """
        This is hashed in an attempt to discover a nonce under the difficulty
        target.
        """
        return (
            f'{self.version}{self.prev_block_hash}{self.merkle_hash}'
            f'{self.timestamp}{self.bits}{nonce or self.nonce}')

    @property
    def id(self) -> str: 
        return Utils.sha256d(self.header())




    def calculate_fees(self, utxo_set:UTXO_Set) -> int:
        """
        Given the txns in a Block, subtract the amount of coin output from the
        inputs. This is kept as a reward by the miner.
        """
        fee = 0

        def utxo_from_block(txin):
            tx = [t.txouts for t in self.txns if t.id == txin.to_spend.txid]
            return tx[0][txin.to_spend.txout_idx] if tx else None

        def find_utxo(txin):
            return utxo_set.utxoSet.get(txin.to_spend) or utxo_from_block(txin)

        for txn in self.txns:
            spent = sum(find_utxo(i).value for i in txn.txins)
            sent = sum(o.value for o in txn.txouts)
            fee += (spent - sent)

        return fee

    @classmethod
    def get_block_subsidy(active_chain) -> int:
        halvings = len(active_chain) // Params.HALVE_SUBSIDY_AFTER_BLOCKS_NUM

        if halvings >= 64:
            return 0

        return 50 * Params.LET_PER_COIN // (2 ** halvings)







