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
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from utils import Utils
from params.Params import Params
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain


import ecdsa
from base58 import b58encode_check


from _thread import RLock

def with_lock(lock):
    def dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return dec

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
        return (
            f'{self.version}{self.prev_block_hash}{self.merkle_hash}'
            f'{self.timestamp}{self.bits}{nonce or self.nonce}')

    @property
    def id(self) -> str: 
        return Utils.sha256d(self.header())


    def calculate_fees(self, utxo_set:UTXO_Set) -> int:

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


    def validate_block(self, active_chain: BlockChain, chain_lock:RLock) -> int:

        @with_lock(chain_lock)
        def get_median_time_past(num_last_blocks: int) -> int:
            """Grep for: GetMedianTimePast."""
            last_n_blocks = active_chain.chain[::-1][:num_last_blocks]

            if not last_n_blocks:
                return 0

            return last_n_blocks[len(last_n_blocks) // 2].timestamp

        if not self.txns:
            raise BlockValidationError('txns empty')

        if self.timestamp - time.time() > Params.MAX_FUTURE_BLOCK_TIME:
            raise BlockValidationError('Block timestamp too far in future')

        if int(self.id, 16) > (1 << (256 - self.bits)):
            raise BlockValidationError("Block header doesn't satisfy bits")

        if [i for (i, tx) in enumerate(self.txns) if tx.is_coinbase] != [0]:
            raise BlockValidationError('First txn must be coinbase and no more')

        try:
            for i, txn in enumerate(self.txns):
                txn.validate_basics(as_coinbase=(i == 0))
        except TxnValidationError:
            logger.exception(f"Transaction {txn} in block {self.id} failed to validate")
            raise BlockValidationError('Invalid txn {txn.id}')

        if MerkleNode.get_merkle_root_of_txns(self.txns).val != self.merkle_hash:
            raise BlockValidationError('Merkle hash invalid')

        if self.timestamp <= get_median_time_past(11):
            raise BlockValidationError('timestamp too old')

        if not self.prev_block_hash and not active_chain:
            # This is the genesis block.
            prev_block_chain_idx = Params.ACTIVE_CHAIN_IDX
        else:
            prev_block, prev_block_height, prev_block_chain_idx = locate_block(
                self.prev_block_hash)

            if not prev_block:
                raise BlockValidationError(
                    f'prev block {self.prev_block_hash} not found in any chain',
                    to_orphan=self)

            # No more validation for a block getting attached to a branch.
            if prev_block_chain_idx != Params.ACTIVE_CHAIN_IDX:
                return prev_block_chain_idx

            # Prev. block found in active chain, but isn't tip => new fork.
            elif prev_block != active_chain[-1]:
                return prev_block_chain_idx + 1  # Non-existent

        if get_next_work_required(block.prev_block_hash) != self.bits:
            raise BlockValidationError('bits is incorrect')

        for txn in self.txns[1:]:
            try:
                txn.validate_txn(siblings_in_block=self.txns[1:],
                             allow_utxo_from_mempool=False)
            except TxnValidationError:
                msg = f"{txn} failed to validate"
                logger.exception(msg)
                raise BlockValidationError(msg)

        return prev_block_chain_idx



    @classmethod
    def get_block_subsidy(height) -> int:
        halvings = height // Params.HALVE_SUBSIDY_AFTER_BLOCKS_NUM

        if halvings >= 64:
            return 0

        return 50 * Params.LET_PER_COIN // (2 ** halvings)







