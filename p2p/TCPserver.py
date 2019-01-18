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


        message = Utils.read_all_from_socket(self.request, gs)
        peer = Peer(self.request.getpeername()[0], int(message.port))


        if not isinstance(message, Message):
            logger.exception('message received is not Message')
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
            logger.exception('received unwanted action request ')



    def locate_block(self, block_hash: str, chain: BlockChain=None) -> (Block, int, int):
        with self.chain_lock:
            chains = [chain] if chain else [self.active_chain, *self.side_branches]

            for chain_idx, chain in enumerate(chains):
                for height, block in enumerate(chain.chain):
                    if block.id == block_hash:
                        return (block, height, chain_idx)
            return (None, None, None)

    def check_block_place(self, block: Block) -> int:
        if self.locate_block(block.id)[0]:
            logger.debug(f'ignore block already seen: {block.id}')
            return None

        try:
            chain_idx = block.validate_block(self.active_chain, self.side_branches, self.chain_lock)
        except BlockValidationError as e:
            logger.exception('block %s failed validation', block.id)
            if e.to_orphan:
                logger.info(f"saw orphan block {block.id}")
                self.orphan_blocks.append(e.to_orphan)
            return None

        # If `validate_block()` returned a non-existent chain index, we're
        # creating a new side branch.
        if chain_idx != Params.ACTIVE_CHAIN_IDX and len(self.side_branches) < chain_idx:
            logger.info(
                f'creating a new side branch (idx {chain_idx}) '
                f'for block {block.id}')
            self.side_branches.append(BlockChain(idx = chain_idx, chain = []))

        return chain_idx

    def handleBlockSyncReq(self, blockid: str, peer: Peer):
        logger.info(f"receive BlockSyncReq from {peer}")
        height = self.locate_block(blockid, self.active_chain)[1] or 1
        with self.chain_lock:
            blocks = self.active_chain.chain[height:(height + Params.CHUNK_SIZE)]

        logger.info(f"sending {len(blocks)} blocks to {peer}")
        if Utils.send_to_peer(Message(Actions.BlocksSyncGet, blocks, Params.PORT_CURRENT), peer):
            if peer not in self.peers:
                self.peers.append(peer)


    def handleBlockSyncGet(self, blocks: Iterable[Block], peer: Peer):
        logger.info(f"recieve BlockSyncGet with {len(blocks)} blocks from {peer}")
        new_blocks = [block for block in blocks if not self.locate_block(block.id)[0]]

        if not new_blocks:
            logger.info('[p2p] initial block download complete')
            self.ibd_done.set()
            return
        with self.chain_lock:
            for block in new_blocks:
                chain_idx  = self.check_block_place(block)

                if chain_idx is not None:
                    if chain_idx == Params.ACTIVE_CHAIN_IDX:
                        if self.active_chain.connect_block(block, self.active_chain, self.side_branches, \
                                                        self.mempool, \
                                        self.utxo_set, self.mine_interrupt, self.peers):
                            Persistence.save_to_disk(self.active_chain)
                    else:
                        self.side_branches[chain_idx-1].chain.append(block)

        new_tip_id = self.active_chain.chain[-1].id
        logger.info(f'continuing initial block download at {new_tip_id}')

        Utils.send_to_peer(Message(Actions.BlocksSyncReq, new_tip_id, Params.PORT_CURRENT), peer)

    def handleTxStatusReq(self, txid: str, peer: Peer):
        def _txn_iterator(chain):
            return (
                (txn, block, height)
                for height, block in enumerate(chain) for txn in block.txns)
        if txid in self.mempool.mempool:
            status = 'txid found in_mempool'
            Utils.send_to_peer(Message(Actions.TxStatusRev, status, Params.PORT_CURRENT), peer)
            return
        for tx, block, height in _txn_iterator(self.active_chain.chain):
            if tx.id == txid:
                status = f'Mined in {block.id} at height {height}'
                Utils.send_to_peer(Message(Actions.TxStatusRev, status, Params.PORT_CURRENT), peer)
                return
        status = f'{txid}:not_found'
        Utils.send_to_peer(Message(Actions.TxStatusRev, status, Params.PORT_CURRENT), peer)

    def handleUTXO4Addr(self, addr: str, peer: Peer):
        utxos4addr = [u for u in self.utxo_set.utxoSet.values() if u.to_address == addr]
        Utils.send_to_peer(Message(Actions.UTXO4AddrRev, utxos4addr, Params.PORT_CURRENT), peer)

    def handleBalance4Addr(self, addr: str, peer: Peer):

        utxos4addr = [u for u in self.utxo_set.utxoSet.values() if u.to_address == addr]
        val = sum(utxo.value for utxo in utxos4addr)
        Utils.send_to_peer(Message(Actions.Balance4AddrRev, val, Params.PORT_CURRENT), peer)

    def handleTxRev(self, txn: Transaction, peer: Peer):
        if isinstance(txn, Transaction):
            if self.mempool.add_txn_to_mempool(txn, self.utxo_set):
                logger.info(f"received txn {txn.id} from peer {peer}")
                for _peer in self.peers:
                    if _peer != peer:
                        Utils.send_to_peer(Message(Actions.TxRev, txn, Params.PORT_CURRENT), _peer)
        else:
            logger.info(f'{txn} is not a Transaction object in handleTxRev')
            return

    def handleBlockRev(self, block: Block, peer: Peer):
        if isinstance(block, Block):
            logger.info(f"received block {block.id} from peer {peer}")
            with self.chain_lock:
                chain_idx  = self.check_block_place(block)
                if chain_idx:

                    if peer not in self.peers:
                        self.peers.append(peer)

                    if chain_idx == Params.ACTIVE_CHAIN_IDX:
                        self.active_chain.connect_block(block)
                        Persistence.save_to_disk(self.active_chain)
                    else:
                        self.side_branches[chain_idx-1].chain.append(block)
                    for _peer in self.peers:
                        if _peer != peer:
                            Utils.send_to_peer(Message(Actions.BlockRev, block, Params.PORT_CURRENT), _peer)

        else:
            logger.info(f'{block} is not a Block')

