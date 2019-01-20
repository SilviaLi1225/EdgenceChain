from ds.Block import Block
from ds.TxIn import TxIn
from ds.MemPool import MemPool
from ds.UTXO_Set import UTXO_Set
from p2p.Peer import Peer
from params.Params import Params
from ds.BlockChain import BlockChain
from utils.Utils import Utils

import logging
import os
import threading

from typing import Iterable

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def connect_block(self, block: Block, active_chain: object, side_branches: Iterable[object],\
                  mempool: MemPool, utxo_set: UTXO_Set, mine_interrupt: threading.Event,\
                  peers: Iterable[Peer], doing_reorg=False) -> bool:

    def _reorg_if_necessary(active_chain: BlockChain, side_branches: Iterable[BlockChain], \
                            mempool: MemPool, utxo_set:UTXO_Set, \
                            mine_interrupt: threading.Event, peers: Iterable[Peer]) -> bool:


        def _try_reorg(branch_idx: int, side_branches: Iterable[BlockChain], active_chain: BlockChain, \
                       fork_height: int, mempool: MemPool, utxo_set:UTXO_Set, \
                       mine_interrupt: threading.Event, peers: Iterable[Peer]) -> bool:

            branch_chain = side_branches[branch_idx - 1]

            fork_block = active_chain.chain[fork_height - 1]

            def disconnect_to_fork(active_chain: BlockChain = active_chain, fork_block: Block = fork_block):
                while active_chain.chain[-1].id != fork_block.id:
                    yield active_chain.disconnect_block(mempool, utxo_set)

            old_active = list(disconnect_to_fork(active_chain, fork_block))[::-1]

            assert branch_chain.chain[0].prev_block_hash == active_chain.chain[-1].id

            def rollback_reorg():

                list(disconnect_to_fork(active_chain, fork_block))

                for block in old_active:
                    assert active_chain.connect_block(block, active_chain, side_branches, mempool, utxo_set, \
                                                      mine_interrupt, peers, \
                                                      doing_reorg=True)

            for block in branch_chain:
                if not active_chain.connect_block(block, active_chain, side_branches, mempool, utxo_set, \
                                                  mine_interrupt, peers, doing_reorg=True):

                    logger.info(f'[ds] reorg of branch {branch_idx} to active_chain failed, decide to rollback')
                    rollback_reorg()
                    return False

            for branch_chain in side_branches:
                if branch_chain.idx == branch_idx:
                    branch_chain.chain = old_active

            logger.info(f'[ds] chain reorg successful with new active_chain height {active_chain.height} and '
                        f'top block id {active_chain.chain[-1].id}')

            return True



        reorged = False
        frozen_side_branches = list(side_branches)

        for _, branch_chain in enumerate(frozen_side_branches):
            branch_idx = branch_chain.idx
            fork_block, fork_height, _ = Block.locate_block(branch_chain.chain[0].prev_block_hash, active_chain)
            active_height = active_chain.height
            branch_height_real = branch_chain.height + fork_height

            if branch_height_real > active_height:
                logger.info(f'[ds] decide to reorg branch {branch_idx} with height {branch_height_real} to \
                    active_chain with real height {active_height}')
                reorged |= _try_reorg(branch_idx, side_branches, active_chain, fork_height, mempool, \
                                     utxo_set, mine_interrupt, peers)

        return reorged



    logger.info(f'[ds] connecting block {block.id} to chain with index: {self.idx}')
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

