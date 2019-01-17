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



from p2p.Peer import Peer
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from ds.MerkleNode import MerkleNode
from ds.BlockChain import BlockChain

import ecdsa
from base58 import b58encode_check
from utils.Utils import Utils
from wallet import Wallet

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


from ds.Block import Block
from ds.MerkleNode import MerkleNode
from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)
from params.Params import Params


def save_to_disk(active_chain: BlockChain, CHAIN_PATH=Params.CHAIN_FILE):
    with open(CHAIN_PATH, "wb") as f:
        logger.info(f"saving chain with {len(active_chain.chain)} blocks")
        f.write(Utils.encode_chain_data(list(active_chain.chain)))


def load_from_disk(active_chain: BlockChain, utxo_set: UTXO_Set, CHAIN_PATH=Params.CHAIN_FILE):


    def _connect_block(block: Block, active_chain: BlockChain, utxo_set: UTXO_Set) -> bool:

        def _validate_block() -> bool:

            def _get_next_work_required(prev_block_hash: str) -> int:
                if not prev_block_hash:
                    return Params.INITIAL_DIFFICULTY_BITS

                prev_block, prev_height = active_chain.chain[-1], len(active_chain.chain)

                if (prev_height + 1) % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:
                    return prev_block.bits

                period_start_block = active_chain[max(
                        prev_height - (Params.DIFFICULTY_PERIOD_IN_BLOCKS - 1), 0)]

                actual_time_taken = prev_block.timestamp - period_start_block.timestamp

                if actual_time_taken < Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
                    # Increase the difficulty
                    return prev_block.bits + 1
                elif actual_time_taken > Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
                    return prev_block.bits - 1
                else:
                    # Wow, that's unlikely.
                    return prev_block.bits

            def _get_median_time_past(num_last_blocks: int) -> int:
                """Grep for: GetMedianTimePast."""
                last_n_blocks = active_chain.chain[::-1][:num_last_blocks]
                if not last_n_blocks:
                    return 0

                return last_n_blocks[len(last_n_blocks) // 2].timestamp

            if not block.txns:
                logger.exception('Loading block with none transactions')
                return False

            if block.timestamp - time.time() > Params.MAX_FUTURE_BLOCK_TIME:
                logger.exception('Block timestamp too far in future')
                return False

            if int(block.id, 16) > (1 << (256 - block.bits)):
                logger.exception("Block header doesn't satisfy bits")
                return False

            if [i for (i, tx) in enumerate(block.txns) if tx.is_coinbase] != [0]:
                logger.exception('First txn must be coinbase and no more')
                return False

            try:
                for i, txn in enumerate(block.txns):
                    txn.validate_basics(as_coinbase=(i == 0))
            except TxnValidationError:
                logger.exception(f"Transaction {txn} in block {block.id} failed to validate")
                return False

            if MerkleNode.get_merkle_root_of_txns(block.txns).val != block.merkle_hash:
                logger.exception('Merkle hash invalid')
                return False

            if block.timestamp <= _get_median_time_past(11):
                logger.exception('timestamp too old')
                return False

            if block.prev_block_hash and block.prev_block_hash != active_chain.chain[-1].id:
                logger.exception('block id is not equal to the prev_block_hash')
                return False

            if _get_next_work_required(block.prev_block_hash) != block.bits:
                logger.exception('bits is incorrect')
                return False

            for txn in block.txns[1:]:
                try:
                    txn.validate_txn(siblings_in_block=block.txns[1:],
                                 allow_utxo_from_mempool=False)
                except TxnValidationError:
                    logger.exception(f"{txn} failed to validate")
                    return False
            return True

        if not _validate_block():
            return False

        logger.info(f'connecting block {block.id} to chain {active_chain.idx}')
        active_chain.chain.append(block)

        for tx in block.txns:
            if not tx.is_coinbase:
                for txin in tx.txins:
                    utxo_set.rm_from_utxo(*txin.to_spend)
            for i, txout in enumerate(tx.txouts):
                utxo_set.add_to_utxo(txout, tx, i, tx.is_coinbase, len(active_chain.chain))

        return True




    if not os.path.isfile(CHAIN_PATH):
        logger.info('chain storage file does not exist')
        return
    else:
        if len(active_chain.chain) > 1:
            logger.exception('more blocks exists when loading chain from disk')
            return
        else:
            active_chain.chain.clear()
    try:
        with open(CHAIN_PATH, "rb") as f:
            msg_len = int(binascii.hexlify(f.read(4) or b'\x00'), 16)
            new_blocks = Utils.deserialize(f.read(msg_len))
            logger.info(f"loading chain from disk with {len(new_blocks)} blocks")
            for block in new_blocks[1:]:
                if not _connect_block(block, active_chain, utxo_set):
                    return
    except Exception:
        logger.exception('failded in loading from chain storage file')
        return








