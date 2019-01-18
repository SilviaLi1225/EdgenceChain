from ds.Block import Block
from ds.TxIn import TxIn
from ds.MemPool import MemPool
from ds.UTXO_Set import UTXO_Set
from p2p.Peer import Peer
from params.Params import Params
from utils.Utils import Utils

import logging
import os
import threading

from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class BlockChain(object):

    def __init__(self, idx: int=0, chain: Iterable[Block]=[]):
        self.index = idx
        self.chain = chain


    @property
    def height(self):
        return len(self.chain)

    @property
    def idx(self):
        return self.index

    def find_txout_for_txin(self, txin: TxIn):

        def _txn_iterator(chain: Iterable[Block]):
            return (
                (txn, block, height)
                for height, block in enumerate(chain) for txn in block.txns)

        txid, txout_idx = txin.to_spend

        for tx, block, height in _txn_iterator(self.chain):
            if tx.id == txid:
                txout = tx.txouts[txout_idx]
                return (txout, tx, txout_idx, tx.is_coinbase, height)


    def disconnect_block(self, mempool: MemPool, utxo_set: UTXO_Set) -> Block:

        block = self.chain[-1]

        for tx in block.txns:
            mempool.mempool[tx.id] = tx

            # Restore UTXO set to what it was before this block.
            for txin in tx.txins:
                if txin.to_spend:  # Account for degenerate coinbase txins.
                    utxo_set.add_to_utxo(*self.find_txout_for_txin(txin))
            for i in range(len(tx.txouts)):
                utxo_set.rm_from_utxo(tx.id, i)

        logger.info(f'block {block.id} disconnected')
        return self.chain.pop()

    # connect_block: block,active_chain, side_branches, mempool, utxo_set, mine_interrupt, peers
    def connect_block(self, block: Block, active_chain: object, side_branches: Iterable[object],\
                      mempool: MemPool, utxo_set:UTXO_Set, mine_interrupt: threading.Event,\
                      peers: Iterable[Peer], doing_reorg=False) -> bool:
        """Accept a block and return the chain index we append it to."""
        # Only exit early on already seen in active_chain when reorging.

        # reorg_if_necessary: active_chain, side_branches, mempool, utxo_set, mine_interrupt, peers
        def _reorg_if_necessary(active_chain: BlockChain, side_branches: Iterable[BlockChain], \
                                mempool: MemPool, utxo_set:UTXO_Set, \
                                mine_interrupt: threading.Event, peers: Iterable[Peer]) -> bool:

            def _try_reorg(branch: BlockChain, branch_idx: int, fork_idx: int, active_chain: BlockChain, \
                           side_branches: Iterable[BlockChain], mempool: MemPool, utxo_set:UTXO_Set, \
                           mine_interrupt: threading.Event, peers: Iterable[Peer]) -> bool:
                # Use the global keyword so that we can actually swap out the reference
                # in case of a reorg.

                fork_block = active_chain[fork_idx]

                def disconnect_to_fork():
                    while active_chain[-1].id != fork_block.id:
                        yield active_chain.disconnect_block(mempool, utxo_set)

                old_active = list(disconnect_to_fork())[::-1]

                assert branch[0].prev_block_hash == active_chain[-1].id

                def rollback_reorg():
                    logger.info(f'reorg of idx {branch_idx} to active_chain failed')
                    list(disconnect_to_fork())  # Force the gneerator to eval.

                    for block in old_active:
                        assert active_chain.connect_block(block, active_chain, side_branches, mempool, utxo_set, \
                                                          mine_interrupt, peers, \
                                                          doing_reorg=True)

                for block in branch:
                    if not active_chain.connect_block(block, active_chain, side_branches, mempool, utxo_set, \
                                                      mine_interrupt, peers, doing_reorg=True):
                        rollback_reorg()
                        return False

                # Fix up side branches: remove new active, add old active.
                side_branches.pop(branch_idx - 1)
                side_branches.append(old_active)

                logger.info(
                    'chain reorg! New height: %s, tip: %s',
                    len(active_chain), active_chain[-1].id)

                return True

            def _locate_block(block_hash: str, chain: BlockChain) -> (Block, int):
                for height, block in enumerate(chain.chain):
                    if block.id == block_hash:
                        return (block, height)
                return (None, None)

            reorged = False
            frozen_side_branches = list(side_branches)  # May change during this call.


            # TODO should probably be using `chainwork` for the basis of comparison here.
            for branch_idx, blockchain in enumerate(frozen_side_branches, 1):
                fork_block, fork_idx = _locate_block(
                    blockchain.chain[0].prev_block_hash, active_chain)
                active_height = active_chain.height
                branch_height = blockchain.height + fork_idx

                if branch_height > active_height:
                    logger.info(
                        f'attempting reorg of idx {branch_idx} to active_chain: '
                        f'new height of {branch_height} (vs. {active_height})')
                    reorged |= _try_reorg(blockchain, branch_idx, fork_idx, active_chain, side_branches, mempool, \
                                         utxo_set, mine_interrupt, peers)

            return reorged



        logger.info(f'connecting block {block.id} to chain {self.idx}')
        self.chain.append(block)
        # If we added to the active chain, perform upkeep on utxo_set and mempool.
        if self.idx == Params.ACTIVE_CHAIN_IDX:
            for tx in block.txns:
                mempool.mempool.pop(tx.id, None)

                if not tx.is_coinbase:
                    for txin in tx.txins:
                        utxo_set.rm_from_utxo(*txin.to_spend)
                for i, txout in enumerate(tx.txouts):
                    utxo_set.add_to_utxo(txout, tx, i, tx.is_coinbase, self.height)

        if (not doing_reorg and \
            _reorg_if_necessary(active_chain, side_branches, mempool, utxo_set, mine_interrupt, peers)) \
                or self.idx == Params.ACTIVE_CHAIN_IDX:
            mine_interrupt.set()


        return True






