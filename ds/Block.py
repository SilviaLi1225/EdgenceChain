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
from utils.Utils import Utils
from params.Params import Params
from ds.MerkleNode import MerkleNode
from ds.TxIn import TxIn
from ds.TxOut import TxOut
from params.Params import Params
import _thread



import ecdsa
from base58 import b58encode_check

from _thread import RLock



logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class Block(NamedTuple):

    version: int
    prev_block_hash: str
    merkle_hash: str
    timestamp: int
    bits: int
    nonce: int
    txns: Iterable[Transaction]

    def header(self, nonce=None) -> str:
        return (
            f'{self.version}{self.prev_block_hash}{self.merkle_hash}'
            f'{self.timestamp}{self.bits}{nonce or self.nonce}')

    @property
    def id(self) -> str: 
        return Utils.sha256d(self.header())

    @classmethod
    def genesis_block(cls):
        return cls(
            version='5465616d3a20456467656e63650a4c65616465723a20776f6c6662726f746865720a4d656d626572733a2063626f7a69'
                    '2c204c6561684c69752c207069616f6c69616e676b622c2053616c7661746f7265303632362c2053696c7669614c69313'
                    '232352c204a69617169204c69752c2078696179756e696c0a',
            prev_block_hash=None,
            merkle_hash='8cfb8d2d2ed9343461b0eefb73c775b9366a32e05e81b0e8946620e2f1935507',
            timestamp=1547747173,
            bits=Params.INITIAL_DIFFICULTY_BITS,
            nonce=9051321,
            txns=[Transaction(
                txins=[TxIn(
                    to_spend=None, unlock_sig=b'0', unlock_pk=None, sequence=0)],
                txouts=[TxOut(
                    value=5000000000, to_address='0000000000000000000000000000000000')], locktime=None)]
        )

    @classmethod
    def get_block_subsidy(cls, active_chain: object) -> int:
        halvings = active_chain.height// Params.HALVE_SUBSIDY_AFTER_BLOCKS_NUM

        if halvings >= 64: return 0

        return 50 * Params.LET_PER_COIN // (2 ** halvings)

    @classmethod
    def locate_block(cls, block_hash, active_chain, side_branches = None):
        chains = [active_chain] if side_branches is None else [active_chain, side_branches]
        for chain_idx, chain in enumerate(chains):
            for height, block in enumerate(chain.chain, 1):
                if block.id == block_hash:
                    if chain_idx != Params.ACTIVE_CHAIN_IDX:
                        fork_height = Block.locate_block(chain.chain[0].prev_block_hash, active_chain)
                        height = fork_height + height
                    return (block, height, chain_idx)
        return (None, None, None)


    @classmethod
    def get_next_work_required(cls, prev_block_hash: str, active_chain: object, \
                               side_branches: Iterable[object] = None) -> int:

        """
        Based on the chain, return the number of difficulty bits the next block
        must solve.
        """
        if not prev_block_hash:
            return Params.INITIAL_DIFFICULTY_BITS

        if side_branches is not None:

            (prev_block, prev_height, pre_chain_idx) = Block.locate_block(prev_block_hash, active_chain, side_branches)

            if pre_chain_idx != 0:
                return None

            if prev_height % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:
                return prev_block.bits


            period_start_block = active_chain.chain[max(
                    prev_height - Params.DIFFICULTY_PERIOD_IN_BLOCKS, 0)]
        elif side_branches is None:
            prev_block, prev_height = active_chain.chain[-1], active_chain.height

            if prev_height % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:
                return prev_block.bits

            period_start_block = active_chain.chain[max(
                    prev_height - Params.DIFFICULTY_PERIOD_IN_BLOCKS, 0)]
        else:
            pass


        actual_time_taken = prev_block.timestamp - period_start_block.timestamp

        if actual_time_taken < Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
            # Increase the difficulty
            return prev_block.bits + 1
        elif actual_time_taken > Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
            return prev_block.bits - 1
        else:
            # Wow, that's unlikely.
            return prev_block.bits



    def calculate_fees(self, utxo_set: UTXO_Set) -> int:

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



    def validate_block(self, active_chain: object, side_branches: Iterable[object] = None) -> int:

        def _get_median_time_past(num_last_blocks: int) -> int:
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
            logger.exception(f"[ds] Transaction {txn} in block {self.id} failed to validate")
            raise BlockValidationError('Invalid txn {txn.id}')

        if MerkleNode.get_merkle_root_of_txns(self.txns).val != self.merkle_hash:
            raise BlockValidationError('Merkle hash invalid')

        if self.timestamp <= _get_median_time_past(11):
            raise BlockValidationError('timestamp too old')

        #
        if not self.prev_block_hash and active_chain.height == 1:
            prev_block_chain_idx = Params.ACTIVE_CHAIN_IDX
        else:
            prev_block, prev_block_height, prev_block_chain_idx = Block.locate_block(
                self.prev_block_hash, active_chain, side_branches)

            if not prev_block:
                raise BlockValidationError(
                    f'prev block {self.prev_block_hash} not found in any chain',
                    to_orphan=self)

            # No more validation for a block getting attached to a branch.
            if prev_block_chain_idx != Params.ACTIVE_CHAIN_IDX:
                return prev_block_chain_idx

            # Prev. block found in active chain, but isn't tip => new fork.
            elif prev_block != active_chain.chain[-1]:
                return prev_block_chain_idx + 1  # Non-existent

        if Block.get_next_work_required(self.prev_block_hash, active_chain, side_branches) != self.bits:
            raise BlockValidationError('bits is incorrect')

        for txn in self.txns[1:]:
            try:
                txn.validate_txn(siblings_in_block=self.txns[1:],
                             allow_utxo_from_mempool=False)
            except TxnValidationError:
                msg = f"[ds] {txn} failed to validate"
                logger.exception(msg)
                raise BlockValidationError(msg)


        return prev_block_chain_idx











