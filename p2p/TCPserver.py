import binascii
import time
import json
import hashlib
import threading
import _thread
import logging
import socketserver
import socket
import random
import os
from functools import lru_cache, wraps
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
from ds.Transaction import Transaction
from ds.Block  import Block
from ds.UnspentTxOut import UnspentTxOut
from utils.Errors import BlockValidationError
from utils.Utils import Utils
from params.Params import Params

from p2p.Message import Message

from p2p.Peer import Peer
from ds.UTXO_Set import UTXO_Set
from ds.MemPool import MemPool
from ds.BlockChain import BlockChain
from ds.TxIn import TxIn
from ds.TxOut import TxOut

from p2p.Message import Message
from p2p.Message import Actions
from persistence import Persistence

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, ip_port, tcp_handler_class, active_chain: BlockChain, side_branches: Iterable[BlockChain], \
                 orphan_blocks: Iterable[Block], utxo_set: UTXO_Set, mempool: MemPool, peers: Iterable[Peer], \
                 mine_interrupt: threading.Event, ibd_done: threading.Event, chain_lock: _thread.RLock):


        self.active_chain = active_chain
        self.side_branches = side_branches
        self.orphan_blocks = orphan_blocks
        self.utxo_set = utxo_set
        self.mempool = mempool
        self.peers = peers
        self.mine_interrupt = mine_interrupt
        self.ibd_done = ibd_done
        self.chain_lock = chain_lock

        socketserver.TCPServer.__init__(self, ip_port, tcp_handler_class)
        logger.info(f'[p2p] listening on {Params.PORT_CURRENT}')

class TCPHandler(socketserver.BaseRequestHandler):


    def handle(self):

        # self.server is an instance of the ThreadedTCPServer
        self.active_chain = self.server.active_chain
        self.side_branches = self.server.side_branches
        self.orphan_blocks = self.server.orphan_blocks
        self.utxo_set = self.server.utxo_set
        self.mempool = self.server.mempool
        self.peers = self.server.peers
        self.mine_interrupt = self.server.mine_interrupt
        self.ibd_done = self.server.ibd_done
        self.chain_lock = self.server.chain_lock




        gs = dict()
        gs['Block'], gs['Transaction'], gs['UnspentTxOut'], gs['Message'], gs['TxIn'], gs['TxOut'] = globals()['Block'], \
                    globals()['Transaction'], globals()['UnspentTxOut'], globals()['Message'], \
                    globals()['TxIn'], globals()['TxOut']
        try:
            message = Utils.read_all_from_socket(self.request, gs)
        except:
            logger.exception(f'[p2p] Invalid meassage from peer {self.request.getpeername()[0]}')
            return


        if not isinstance(message, Message):
            logger.exception(f'[p2p] Not a Message from peer {self.request.getpeername()[0]}')
            return
        else:
            peer = Peer(str(self.request.getpeername()[0]), int(message.port))
            if (peer.ip == '127.0.0.1' and peer.port == Params.PORT_CURRENT) or \
                (peer.ip == 'localhost' and peer.port == Params.PORT_CURRENT):
                logger.exception(f'[p2p] new found {peer} is the current node itself, and does nothing for it')
                return




        action = int(message.action)
        if action == Actions.BlocksSyncReq:
            self.handleBlockSyncReq(message.data, peer)
        elif action == Actions.BlocksSyncGet:
            self.handleBlockSyncGet(message.data, peer)
        elif action == Actions.TxStatusReq:
            self.handleTxStatusReq(message.data, peer)
        elif action == Actions.UTXO4Addr:
            self.handleUTXO4Addr(message.data, peer)
        elif action == Actions.Balance4Addr:
            self.handleBalance4Addr(message.data, peer)
        elif action == Actions.TxRev:
            self.handleTxRev(message.data, peer)
        elif action == Actions.BlockRev:
            self.handleBlockRev(message.data, peer)
        else:
            logger.exception(f'[p2p] received unwanted action request ')

    def handleBlockSyncReq(self, blockid: str, peer: Peer):
        with self.chain_lock:
            height = Block.locate_block(blockid, self.active_chain)[1]
            if height is None:
                logger.info(f'[p2p] cannot find blockid {blockid}, and do nothing for this BlockSyncReq from peer {peer}')
                return
            else:
                logger.info(f"[p2p] receive BlockSyncReq at height {height} from peer {peer}")
            blocks = self.active_chain.chain[height:(height + Params.CHUNK_SIZE)]

        logger.info(f"[p2p] sending {len(blocks)} blocks to {peer}")
        if Utils.send_to_peer(Message(Actions.BlocksSyncGet, blocks, Params.PORT_CURRENT), peer):
            if peer not in self.peers:
                self.peers.append(peer)
                logger.info(f'[p2p] add peer {peer} into peer list')
                Peer.save_peers(self.peers)

    def handleBlockSyncGet(self, blocks: Iterable[Block], peer: Peer):
        logger.info(f"[p2p] recieve BlockSyncGet with {len(blocks)} blocks from {peer}")
        new_blocks = [block for block in blocks if not Block.locate_block(block.id, self.active_chain, self.side_branches)[0]]

        if not new_blocks:
            logger.info('[p2p] initial block download complete, prepare to mine')
            self.ibd_done.set()
            return
        else:
            self.ibd_done.clear()

        with self.chain_lock:
            for block in new_blocks:
                chain_idx  = TCPHandler.check_block_place(block, self.active_chain, self.side_branches, self.chain_lock)

                if chain_idx is not None and chain_idx >= 0:
                    TCPHandler.do_connect_block_and_after(block, chain_idx, self.active_chain, self.side_branches, \
                                                          self.mempool, \
                                                           self.utxo_set, self.mine_interrupt, self.peers)
                elif chain_idx <= -1:
                    logger.info(f'[p2p] orphan or wrong block {block.id}')
                    break
                else:
                    logger.info(f'[p2p] do nothing for block {block.id}')


            new_tip_id = self.active_chain.chain[-1].id
        logger.info(f'[p2p] current chain height {self.active_chain.height}, and continue initial block download ... ')

        Utils.send_to_peer(Message(Actions.BlocksSyncReq, new_tip_id, Params.PORT_CURRENT), peer)

    def handleTxStatusReq(self, txid: str, peer: Peer):
        def _txn_iterator(chain):
            return (
                (txn, block, height)
                for height, block in enumerate(chain, 1) for txn in block.txns)
        with self.chain_lock:
            if txid in self.mempool.mempool:
                status = f'txn {txid} found in_mempool'
                Utils.send_to_peer(Message(Actions.TxStatusRev, status, Params.PORT_CURRENT), peer)
                return
            for tx, block, height in _txn_iterator(self.active_chain.chain):
                if tx.id == txid:
                    status = f'txn {txid} is mined in block {block.id} at height {height}'
                    Utils.send_to_peer(Message(Actions.TxStatusRev, status, Params.PORT_CURRENT), peer)
                    return
        status = f'txn {txid}:not_found'
        Utils.send_to_peer(Message(Actions.TxStatusRev, status, Params.PORT_CURRENT), peer)

    def handleUTXO4Addr(self, addr: str, peer: Peer):
        with self.chain_lock:
            utxos4addr = [u for u in self.utxo_set.utxoSet.values() if u.to_address == addr]
        Utils.send_to_peer(Message(Actions.UTXO4AddrRev, utxos4addr, Params.PORT_CURRENT), peer)

    def handleBalance4Addr(self, addr: str, peer: Peer):

        with self.chain_lock:
            utxos4addr = [u for u in self.utxo_set.utxoSet.values() if u.to_address == addr]
        val = sum(utxo.value for utxo in utxos4addr)
        Utils.send_to_peer(Message(Actions.Balance4AddrRev, val, Params.PORT_CURRENT), peer)

    def handleTxRev(self, txn: Transaction, peer: Peer):
        if isinstance(txn, Transaction):
            logger.info(f"[p2p] received txn {txn.id} from peer {peer}")
            with self.chain_lock:
                if self.mempool.add_txn_to_mempool(txn, self.utxo_set):
                    for _peer in self.peers:
                        if _peer != peer:
                            Utils.send_to_peer(Message(Actions.TxRev, txn, Params.PORT_CURRENT), _peer)
        else:
            logger.info(f'[p2p] {txn} is not a Transaction object in handleTxRev')
            return

    def handleBlockRev(self, block: Block, peer: Peer):
        if isinstance(block, Block):
            logger.info(f"[p2p] received block {block.id} from peer {peer}")
            with self.chain_lock:
                chain_idx  = TCPHandler.check_block_place(block, self.active_chain, self.side_branches, self.chain_lock)
                if chain_idx is not None and chain_idx >= 0:
                    TCPHandler.do_connect_block_and_after(block, chain_idx, self.active_chain, self.side_branches, \
                                                       self.mempool, self.utxo_set, self.mine_interrupt, self.peers)
                elif chain_idx is None:
                    logger.info(f'[p2p] already seen block {block.id}, and do nothing')
                elif chain_idx == -1:
                    self.orphan_blocks.append(block)

        else:
            logger.info(f'[p2p] {block} is not a Block')

    @classmethod
    def do_connect_block_and_after(cls, block, chain_idx, active_chain: BlockChain, side_branches: Iterable[BlockChain], \
                                mempool: MemPool, utxo_set:UTXO_Set, mine_interrupt: threading.Event, \
                                peers: Iterable[Peer]):
        if chain_idx == Params.ACTIVE_CHAIN_IDX:
            connect_block_success =  active_chain.connect_block(block, active_chain, \
                                                    side_branches, \
                                    mempool, utxo_set, mine_interrupt, peers)
        else:
            connect_block_success =  side_branches[chain_idx-1].connect_block(block, \
                                             active_chain, side_branches, \
                                    mempool, utxo_set, mine_interrupt, peers)
        if connect_block_success is True:
            Persistence.save_to_disk(active_chain)
            side_branches_to_discard = []
            for branch_chain in side_branches:
                fork_block, fork_height, _ = Block.locate_block(branch_chain.chain[0].prev_block_hash,
                                                        active_chain)
                branch_height_real = branch_chain.height + fork_height
                if active_chain.height - branch_height_real > Params.MAXIMUM_ALLOWABLE_HEIGHT_DIFF:
                    side_branches_to_discard.append(branch_chain)
            for branch_chain in side_branches_to_discard:
                side_branches.remove(branch_chain)
            for index, branch_chain in enumerate(side_branches, 1):
                branch_chain.index = index

    @classmethod
    def check_block_place(cls, block: Block, active_chain: BlockChain, side_branches: Iterable[BlockChain], \
                          chain_lock: _thread.RLock) -> int:
        if Block.locate_block(block.id, active_chain, side_branches)[0]:
            logger.debug(f'[p2p] ignore block already seen: {block.id}')
            return None # already seen block

        try:
            chain_idx = block.validate_block(active_chain, side_branches, chain_lock)
        except BlockValidationError as e:
            logger.exception('block %s failed validation', block.id)
            if e.to_orphan:
                logger.info(f"[p2p] saw orphan block {block.id}")
                return -1  # orphan block
            else:
                return -2  # Internal error in this block

        with chain_lock:
            if chain_idx != Params.ACTIVE_CHAIN_IDX and len(side_branches) < chain_idx:
                logger.info(
                    f'[p2p] creating a new side branch (idx {chain_idx}) '
                    f'for block {block.id}')
                side_branches.append(BlockChain(idx = chain_idx, chain = []))

        return chain_idx
