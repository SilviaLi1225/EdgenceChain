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

from dataStructure.DataStructure  import (OutPoint, TxIn, TxOut, UnspentTxOut, Transaction,
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



# Used to represent the specific output within a transaction.
OutPoint = NamedTuple('OutPoint', [('txid', str), ('txout_idx', int)])


class TxIn(NamedTuple):
    """Inputs to a Transaction."""
    # A reference to the output we're spending. This is None for coinbase
    # transactions.
    to_spend: Union[OutPoint, None]

    # The (signature, pubkey) pair which unlocks the TxOut for spending.
    unlock_sig: bytes
    unlock_pk: bytes

    # A sender-defined sequence number which allows us replacement of the txn
    # if desired.
    sequence: int


class TxOut(NamedTuple):
    """Outputs from a Transaction."""
    # The number of LET this awards.
    value: int

    # The public key of the owner of this Txn.
    to_address: str


class UnspentTxOut(NamedTuple):
    value: int
    to_address: str

    # The ID of the transaction this output belongs to.
    txid: str
    txout_idx: int

    # Did this TxOut from from a coinbase transaction?
    is_coinbase: bool

    # The blockchain height this TxOut was included in the chain.
    height: int

    @property
    def outpoint(self): return OutPoint(self.txid, self.txout_idx)


class Transaction(NamedTuple):
    txins: Iterable[TxIn]
    txouts: Iterable[TxOut]

    # The block number or timestamp at which this transaction is unlocked.
    # < 500000000: Block number at which this transaction is unlocked.
    # >= 500000000: UNIX timestamp at which this transaction is unlocked.
    locktime: int = None

    @property
    def is_coinbase(self) -> bool:
        return len(self.txins) == 1 and self.txins[0].to_spend is None

    @classmethod
    def create_coinbase(cls, pay_to_addr, value, height):
        return cls(
            txins=[TxIn(
                to_spend=None,
                # Push current block height into unlock_sig so that this
                # transaction's ID is unique relative to other coinbase txns.
                unlock_sig=str(height).encode(),
                unlock_pk=None,
                sequence=0)],
            txouts=[TxOut(
                value=value,
                to_address=pay_to_addr)],
        )

    @property
    def id(self) -> str:
        return sha256d(serialize(self))

    def validate_basics(self, as_coinbase=False):
        if (not self.txouts) or (not self.txins and not as_coinbase):
            raise TxnValidationError('Missing txouts or txins')

        if len(serialize(self)) > Params.MAX_BLOCK_SERIALIZED_SIZE:
            raise TxnValidationError('Too large')

        if sum(t.value for t in self.txouts) > Params.MAX_MONEY:
            raise TxnValidationError('Spend value too high')

	@classmethod
	def validate_txn(cls,
					 as_coinbase: bool = False,
					 siblings_in_block: Iterable[Transaction] = None,
					 allow_utxo_from_mempool: bool = True,
					 ) -> bool:
		"""
		Validate a single transaction. Used in various contexts, so the
		parameters facilitate different uses.
		"""
		cls.validate_basics(as_coinbase=as_coinbase)

		available_to_spend = 0

		for i, txin in enumerate(txn.txins):
			utxo = utxo_set.get(txin.to_spend)

			if siblings_in_block:
				utxo = utxo or find_utxo_in_list(txin, siblings_in_block)

			if allow_utxo_from_mempool:
				utxo = utxo or find_utxo_in_mempool(txin)

			if not utxo:
				raise TxnValidationError(
					f'Could find no UTXO for TxIn[{i}] -- orphaning txn',
					to_orphan=cls)

			if utxo.is_coinbase and \
					(get_current_height() - utxo.height) < \
					Params.COINBASE_MATURITY:
				raise TxnValidationError(f'Coinbase UTXO not ready for spend')

			try:
				validate_signature_for_spend(txin, utxo, cls)
			except TxUnlockError:
				raise TxnValidationError(f'{txin} is not a valid spend of {utxo}')

			available_to_spend += utxo.value

		if available_to_spend < sum(o.value for o in cls.txouts):
			raise TxnValidationError('Spend value is more than available')

		return True

	def validate_signature_for_spend(txin, utxo: UnspentTxOut, txn):
	    pubkey_as_addr = pubkey_to_address(txin.unlock_pk)
	    verifying_key = ecdsa.VerifyingKey.from_string(
		txin.unlock_pk, curve=ecdsa.SECP256k1)

	    if pubkey_as_addr != utxo.to_address:
		raise TxUnlockError("Pubkey doesn't match")

	    try:
		spend_msg = build_spend_message(
		    txin.to_spend, txin.unlock_pk, txin.sequence, txn.txouts)
		verifying_key.verify(txin.unlock_sig, spend_msg)
	    except Exception:
		logger.exception('Key verification failed')
		raise TxUnlockError("Signature doesn't match")

	    return True


	def build_spend_message(to_spend, pk, sequence, txouts) -> bytes:
	    """This should be ~roughly~ equivalent to SIGHASH_ALL."""
	    return sha256d(
		serialize(to_spend) + str(sequence) +
		binascii.hexlify(pk).decode() + serialize(txouts)).encode()










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
    	return sha256d(self.header())




    def calculate_fees(self) -> int:
	"""
	Given the txns in a Block, subtract the amount of coin output from the
	inputs. This is kept as a reward by the miner.
	"""
	fee = 0

	def utxo_from_block(txin):
	    tx = [t.txouts for t in self.txns if t.id == txin.to_spend.txid]
	    return tx[0][txin.to_spend.txout_idx] if tx else None

	def find_utxo(txin):
	    return utxo_set.get(txin.to_spend) or utxo_from_block(txin)

	for txn in block.txns:
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










# Merkle trees
# ----------------------------------------------------------------------------

class MerkleNode(NamedTuple):
    val: str
    children: Iterable = None

    @classmethod
	def get_merkle_root_of_txns(cls, txns):
	    return cls.get_merkle_root(*[t.id for t in txns])

    @classmethod
    @lru_cache(maxsize=1024)
    def get_merkle_root(cls, *leaves: Tuple[str]) -> cls:
	"""Builds a Merkle tree and returns the root given some leaf values."""
	if len(leaves) % 2 == 1:
	    leaves = leaves + (leaves[-1],)

	def _chunks(l, n) -> Iterable[Iterable]:
	    return (l[i:i + n] for i in range(0, len(l), n))

	def find_root(nodes):
	    newlevel = [
		cls(sha256d(i1.val + i2.val), children=[i1, i2])
		for [i1, i2] in _chunks(nodes, 2)
	    ]

	    return find_root(newlevel) if len(newlevel) > 1 else newlevel[0]

	return find_root([cls(sha256d(l)) for l in leaves])



  # UTXO set
# ----------------------------------------------------------------------------
#utxo_set = UTXO_Set() #然后对utxo_set的操作就转化为对utxo_set对象及其utxoSet成员的操作

class UTXO_Set(object):

    def __init__(self):
    	self.utxoSet: Mapping[OutPoint, UnspentTxOut] = {}


	def add_to_utxo(self, txout, tx, idx, is_coinbase, height):
	    utxo = UnspentTxOut(
	        *txout,
	        txid=tx.id, txout_idx=idx, is_coinbase=is_coinbase, height=height)

	    logger.info(f'adding tx outpoint {utxo.outpoint} to utxo_set')
	    self.utxoSet[utxo.outpoint] = utxo


	def rm_from_utxo(self, txid, txout_idx):
	    del self.utxoSet[OutPoint(txid, txout_idx)]

    @classmethod  
    def find_utxo_in_list(cls, txin, txns) -> UnspentTxOut:
	txid, txout_idx = txin.to_spend
	try:
	    txout = [t for t in txns if t.id == txid][0].txouts[txout_idx]
	except Exception:
	    return None

	return UnspentTxOut(
	    *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=txout_idx)

# mempool  Set of yet-unmined transactions.

class MemPool(object): 
    def __init__(self):
    	self.mempool: Dict[str, Transaction] = {}

		# Set of orphaned (i.e. has inputs referencing yet non-existent UTXOs)
		# transactions.
		self.orphan_txns: Iterable[Transaction] = []


	def find_utxo_in_mempool(self,txin) -> UnspentTxOut:
	    txid, idx = txin.to_spend

	    try:
	        txout = self.mempool[txid].txouts[idx]
	    except Exception:
	        logger.debug("Couldn't find utxo in mempool for %s", txin)
	        return None

	    return UnspentTxOut(
	        *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=idx)


	def select_from_mempool(self, block: Block) -> Block:
	    """Fill a Block with transactions from the mempool."""
	    added_to_block = set()

	    def check_block_size(block) -> bool:
	        return len(serialize(block)) < Params.MAX_BLOCK_SERIALIZED_SIZE

	    def try_add_to_block(block, txid) -> Block:
	        if txid in added_to_block:
	            return block

	        tx = self.mempool[txid]

	        # For any txin that can't be found in the main chain, find its
	        # transaction in the mempool (if it exists) and add it to the block.

	        for txin in tx.txins:
	            if txin.to_spend in utxo_set:
	                continue

	            in_mempool = self.find_utxo_in_mempool(txin)

	            if not in_mempool:
	                logger.debug(f"Couldn't find UTXO for {txin}")
	                return None

	            block = try_add_to_block(block, in_mempool.txid)
	            if not block:
	                logger.debug(f"Couldn't add parent")
	                return None

	        newblock = block._replace(txns=[*block.txns, tx])

	        if check_block_size(newblock):
	            logger.debug(f'added tx {tx.id} to block')
	            added_to_block.add(txid)
	            return newblock
	        else:
	            return block

	    for txid in self.mempool:
	        newblock = try_add_to_block(block, txid)

	        if check_block_size(newblock):
	            block = newblock
	        else:
	            break

	    return block


	def add_txn_to_mempool(self, txn: Transaction):
	    if txn.id in mempool:
	        logger.info(f'txn {txn.id} already seen')
	        return

	    try:
	        txn = validate_txn(txn)
	    except TxnValidationError as e:
	        if e.to_orphan:
	            logger.info(f'txn {e.to_orphan.id} submitted as orphan')
	            self.orphan_txns.append(e.to_orphan)
	        else:
	            logger.exception(f'txn rejected')
	    else:
	        logger.info(f'txn {txn.id} added to mempool')
	        self.mempool[txn.id] = txn

	        for peer in peer_hostnames:
	            send_to_peer(txn, peer)

  
