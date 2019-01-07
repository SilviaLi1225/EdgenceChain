#!/usr/bin/env python3
"""
⛼  edgencechain

  putting the rough in "rough consensus"


Some terminology:

- Chain: an ordered list of Blocks, each of which refers to the last and
      cryptographically preserves a history of Transactions.

- Transaction (or tx or txn): a list of inputs (i.e. past outputs being spent)
    and outputs which declare value assigned to the hash of a public key.

- PoW (proof of work): the solution to a puzzle which allows the acceptance
    of an additional Block onto the chain.

- Reorg: chain reorganization. When a side branch overtakes the main chain.


An incomplete list of unrealistic simplifications:

- Byte encoding and endianness are very important when serializing a
  data structure to be hashed in Bitcoin and are not reproduced
  faithfully here. In fact, serialization of any kind here is slipshod and
  in many cases relies on implicit expectations about Python JSON
  serialization.

- Transaction types are limited to P2PKH.

- Initial Block Download eschews `getdata` and instead returns block payloads
  directly in `inv`.

- Peer "discovery" is done through environment variable hardcoding. In
  bitcoin core, this is done with DNS seeds.
  See https://bitcoin.stackexchange.com/a/3537/56368


Resources:

- https://en.bitcoin.it/wiki/Protocol_rules
- https://en.bitcoin.it/wiki/Protocol_documentation
- https://bitcoin.org/en/developer-guide
- https://github.com/bitcoinbook/bitcoinbook/blob/second_edition/ch06.asciidoc


TODO:

- deal with orphan blocks              #额。。。
- keep the mempool heap sorted by fee  # 与效率有关，非功能性需求，故暂不考虑
- make use of Transaction.locktime     # 后边会有用到
? make use of TxIn.sequence; i.e. replace-by-fee  # 与locktime结合来用，暂不考虑

"""
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

import ecdsa
from base58 import b58encode_check


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class Params:
    # The infamous max block size.
    MAX_BLOCK_SERIALIZED_SIZE = 1000000  # bytes = 1MB

    # Coinbase transaction outputs can be spent after this many blocks have
    # elapsed since being mined.
    #
    # This is "100" in bitcoin core.
    COINBASE_MATURITY = 2  #2个区块后，Coinbase交易可以花掉

    # Accept blocks timestamped as being from the future, up to this amount.
    MAX_FUTURE_BLOCK_TIME = (60 * 60 * 2)

    # The number of Lets per coin. #realname COIN
    LET_PER_COIN = int(100e6) 

    TOTAL_COINS = 21_000_000

    # The maximum number of Lets that will ever be found.
    MAX_MONEY = LET_PER_COIN * TOTAL_COINS

    # The duration we want to pass between blocks being found, in seconds.
    # This is lower than Bitcoin's configuation (10 * 60).
    #
    # #realname PowTargetSpacing
    TIME_BETWEEN_BLOCKS_IN_SECS_TARGET = 1 * 60

    # The number of seconds we want a difficulty period to last.
    #
    # Note that this differs considerably from the behavior in Bitcoin, which
    # is configured to target difficulty periods of (10 * 2016) minutes.
    #
    # #realname PowTargetTimespan
    DIFFICULTY_PERIOD_IN_SECS_TARGET = (60 * 60 * 10)

    # After this number of blocks are found, adjust difficulty.
    #
    # #realname DifficultyAdjustmentInterval
    DIFFICULTY_PERIOD_IN_BLOCKS = (
        DIFFICULTY_PERIOD_IN_SECS_TARGET / TIME_BETWEEN_BLOCKS_IN_SECS_TARGET)

    # The number of right-shifts applied to 2 ** 256 in order to create the
    # initial difficulty target necessary for mining a block.
    INITIAL_DIFFICULTY_BITS = 24

    # The number of blocks after which the mining subsidy will halve.
    #
    # #realname SubsidyHalvingInterval
    HALVE_SUBSIDY_AFTER_BLOCKS_NUM = 210_000


# Used to represent the specific output within a transaction. 
OutPoint = NamedTuple('OutPoint', [('txid', str), ('txout_idx', int)])  # UTXO所在的交易ID和检索号


class TxIn(NamedTuple):
    """Inputs to a Transaction."""
    # A reference to the output we're spending. This is None for coinbase
    # transactions.
    to_spend: Union[OutPoint, None] #to_spend或者取值一个OutPoint实例，要么取值None，后者表示所在交易将是Coinbase交易。

    # The (signature, pubkey) pair which unlocks the TxOut for spending.
    unlock_sig: bytes   #TxIn作为交易的一个输入，其背后是一个UTXO，因此必须有该UTXO输入的解锁脚本
    unlock_pk: bytes

    # A sender-defined sequence number which allows us replacement of the txn
    # if desired.
    sequence: int # 注意不是Transaction类里的locktime。


class TxOut(NamedTuple): # 目前只用了to_address来锁定，因此只能实现P2PK交易
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

    tmp_1 : int
    tmp_2 : int = 0
    tmp_3 : str = ''
    # The block number or timestamp at which this transaction is unlocked.
    # < 500000000: Block number at which this transaction is unlocked.
    # >= 500000000: UNIX timestamp at which this transaction is unlocked.
    locktime: int = None # 虽然定义了locktime，但目前版本并未用到该值

    @property
    def is_coinbase(self) -> bool:
        return len(self.txins) == 1 and self.txins[0].to_spend is None #只有一个输入，且该输入背后的UTXO（即to_spend属性）为空。

    @classmethod  # 类方法，可以被类或对象调用
    def create_coinbase(cls, pay_to_addr, value, height):
        return cls(     #返回当前类的一个实例，即Transation实例
            txins=[TxIn(
                to_spend=None,
                # Push current block height into unlock_sig so that this
                # transaction's ID is unique relative to other coinbase txns.
                unlock_sig=str(height).encode(),  #随意制定，只要该交易的hash与之前的coinbase交易不重复即可
                unlock_pk=None,
                sequence=0)],
            txouts=[TxOut(
                value=value,
                to_address=pay_to_addr)],
            tmp1 = 1
        )

    @property
    def id(self) -> str: #交易实例的hash
        return sha256d(serialize(self))

    def validate_basics(self, as_coinbase=False):
        if (not self.txouts) or (not self.txins and not as_coinbase): # 交易的输出或输入不存在
            raise TxnValidationError('Missing txouts or txins')

        if len(serialize(self)) > Params.MAX_BLOCK_SERIALIZED_SIZE: #该交易数据的字节数太大
            raise TxnValidationError('Too large')

        if sum(t.value for t in self.txouts) > Params.MAX_MONEY:  
            raise TxnValidationError('Spend value too high')


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
    def id(self) -> str: return sha256d(self.header())


# Chain
# ----------------------------------------------------------------------------

genesis_block = Block(
    version=0, prev_block_hash=None,
    merkle_hash=(
        '7118894203235a955a908c0abfc6d8fe6edec47b0a04ce1bf7263da3b4366d22'),
    timestamp=1501821412, bits=24, nonce=10126761,
    txns=[Transaction(
        txins=[TxIn(
            to_spend=None, unlock_sig=b'0', unlock_pk=None, sequence=0)],
        txouts=[TxOut(
            value=5000000000,
            to_address='143UVyz7ooiAv1pMqbwPPpnH4BV9ifJGFF')], locktime=None,
        tmp_1 = 1)])

# The highest proof-of-work, valid blockchain.
#
# #realname chainActive
active_chain: Iterable[Block] = [genesis_block]

# Branches off of the main chain.
side_branches: Iterable[Iterable[Block]] = []

# Synchronize access to the active chain and side branches.
chain_lock = threading.RLock()


def with_lock(lock):  #装饰器本身带参数的情况，使用时用@with_lock(chain_lock)
    def dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs): #参数定义是(*args, **kw)，表示可以接受任意参数的调用
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return dec


orphan_blocks: Iterable[Block] = []  # 目前的版本里，只是实现了识别orphan_block并添加进该列表（connect_block函数），没有后续处理

# Used to signify the active chain in `locate_block`.
ACTIVE_CHAIN_IDX = 0


@with_lock(chain_lock)
def get_current_height(): return len(active_chain)  # 主链的长度


@with_lock(chain_lock)
def txn_iterator(chain):  #枚举链chain的所有交易，而且对每个交易给出其所在block和height
    return (
        (txn, block, height)
        for height, block in enumerate(chain) for txn in block.txns)


@with_lock(chain_lock)
def locate_block(block_hash: str, chain=None) -> (Block, int, int): #根据区块hash给出所在的链chain_idx，区块block及高度height
    chains = [chain] if chain else [active_chain, *side_branches]

    for chain_idx, chain in enumerate(chains):
        for height, block in enumerate(chain):
            if block.id == block_hash:
                return (block, height, chain_idx)
    return (None, None, None)


@with_lock(chain_lock)
def connect_block(block: Union[str, Block],
                  doing_reorg=False,
                  ) -> Union[None, Block]:
    """Accept a block and return the chain index we append it to."""
    # Only exit early on already seen in active_chain when reorging.
    search_chain = active_chain if doing_reorg else None

    if locate_block(block.id, chain=search_chain)[0]: #如果在某条链上发现了该区块已经存在
        logger.debug(f'ignore block already seen: {block.id}')
        return None

    try:
        block, chain_idx = validate_block(block)  #验证该区块，并返回该区块所属的链
    except BlockValidationError as e:
        logger.exception('block %s failed validation', block.id)
        if e.to_orphan:
            logger.info(f"saw orphan block {block.id}")
            orphan_blocks.append(e.to_orphan)
        return None

    # If `validate_block()` returned a non-existent chain index, we're
    # creating a new side branch.
    if chain_idx != ACTIVE_CHAIN_IDX and len(side_branches) < chain_idx:
        logger.info(
            f'creating a new side branch (idx {chain_idx}) '
            f'for block {block.id}')
        side_branches.append([])

    logger.info(f'connecting block {block.id} to chain {chain_idx}')
    chain = (active_chain if chain_idx == ACTIVE_CHAIN_IDX else
             side_branches[chain_idx - 1])
    chain.append(block)

    # If we added to the active chain, perform upkeep on utxo_set and mempool.
    if chain_idx == ACTIVE_CHAIN_IDX:   #更新UTXO和未打包的交易集合mempool
        for tx in block.txns:
            mempool.pop(tx.id, None)

            if not tx.is_coinbase:
                for txin in tx.txins:
                    rm_from_utxo(*txin.to_spend)  #此处*表示解压参数
            for i, txout in enumerate(tx.txouts):
                add_to_utxo(txout, tx, i, tx.is_coinbase, len(chain))

    if (not doing_reorg and reorg_if_necessary()) or \
            chain_idx == ACTIVE_CHAIN_IDX:
        mine_interrupt.set()
        logger.info(
            f'block accepted '
            f'height={len(active_chain) - 1} txns={len(block.txns)}')

    for peer in peer_hostnames:
        send_to_peer(block, peer)

    return chain_idx


@with_lock(chain_lock)
def disconnect_block(block, chain=None): # 相当于删掉一个区块，所以要恢复该区块消耗掉的UTXO和交易，删掉该区块产生的UTXO
    chain = chain or active_chain
    assert block == chain[-1], "Block being disconnected must be tip." #tip有尖端之意，即要摘掉的是链的最后一个区块

    for tx in block.txns:
        mempool[tx.id] = tx

        # Restore UTXO set to what it was before this block.
        for txin in tx.txins:
            if txin.to_spend:  # Account for degenerate coinbase txins.
                add_to_utxo(*find_txout_for_txin(txin, chain))
        for i in range(len(tx.txouts)):
            rm_from_utxo(tx.id, i)

    logger.info(f'block {block.id} disconnected')
    return chain.pop()


def find_txout_for_txin(txin, chain):
    txid, txout_idx = txin.to_spend

    for tx, block, height in txn_iterator(chain):
        if tx.id == txid:
            txout = tx.txouts[txout_idx]
            return (txout, tx, txout_idx, tx.is_coinbase, height)


@with_lock(chain_lock)
def reorg_if_necessary() -> bool:
    reorged = False
    frozen_side_branches = list(side_branches)  # May change during this call.

    # TODO should probably be using `chainwork` for the basis of
    # comparison here.
    for branch_idx, chain in enumerate(frozen_side_branches, 1):  #列举各个分支链chain，序号branch_idx从1开始
        fork_block, fork_idx, _ = locate_block(
            chain[0].prev_block_hash, active_chain) #定位该分支链的分叉区块位置，返回该区块fork_block及其高度fork_idx
        active_height = len(active_chain)  #主链的高度
        branch_height = len(chain) + fork_idx  #分支链chain的高度

        if branch_height > active_height: #替换主链的条件
            logger.info(
                f'attempting reorg of idx {branch_idx} to active_chain: '
                f'new height of {branch_height} (vs. {active_height})')
            reorged |= try_reorg(chain, branch_idx, fork_idx) # a|=b等价于a = a|b

    return reorged


@with_lock(chain_lock)
def try_reorg(branch, branch_idx, fork_idx) -> bool:
    # Use the global keyword so that we can actually swap out the reference
    # in case of a reorg.
    global active_chain  #对全局变量active_chain和side_branches进行了赋值操作，即指针发生改变，因此需要声明，不然会认为是局部变量
    global side_branches

    fork_block = active_chain[fork_idx]

    def disconnect_to_fork():
        while active_chain[-1].id != fork_block.id: # 区块的id，就是其区块hash值
            yield disconnect_block(active_chain[-1])

    old_active = list(disconnect_to_fork())[::-1] # 列出主链上分叉之后的区块，并按时间顺序排列

    assert branch[0].prev_block_hash == active_chain[-1].id  #确保disconnect_to_fork函数执行正确

    def rollback_reorg(): #失败，回滚
        logger.info(f'reorg of idx {branch_idx} to active_chain failed')
        list(disconnect_to_fork())  # Force the generator to eval.  

        for block in old_active: # 将摘出来的区块，再按与按顺序接上去
            assert connect_block(block, doing_reorg=True) == ACTIVE_CHAIN_IDX

    for block in branch:
        connected_idx = connect_block(block, doing_reorg=True) #将分叉链上的区块，一个个接上去
        if connected_idx != ACTIVE_CHAIN_IDX:  #疑问：如果在中途有此问题，怎么办？莫非这种情况不会发生？
            rollback_reorg()
            return False

    # Fix up side branches: remove new active, add old active.
    side_branches.pop(branch_idx - 1)  
    side_branches.append(old_active)

    logger.info(
        'chain reorg! New height: %s, tip: %s',
        len(active_chain), active_chain[-1].id)

    return True


def get_median_time_past(num_last_blocks: int) -> int:
    """Grep for: GetMedianTimePast."""
    last_n_blocks = active_chain[::-1][:num_last_blocks]

    if not last_n_blocks:
        return 0

    return last_n_blocks[len(last_n_blocks) // 2].timestamp # 默认的num_last_blocks是11，而11//2=5


# Chain Persistance
# ----------------------------------------------------------------------------

CHAIN_PATH = os.environ.get('TC_CHAIN_PATH', 'chain.dat') #区块链数据的存储路径，不预先指定，则默认为当前目录下的chain.dat文件

@with_lock(chain_lock)
def save_to_disk():
    with open(CHAIN_PATH, "wb") as f:  #打开文件的参数是wb，原有内容会被删除
        logger.info(f"saving chain with {len(active_chain)} blocks")
        f.write(encode_socket_data(list(active_chain)))   #把主链的区块数据一次传输并写进文件，前面四个字节是数据长度，其后是数据

        
@with_lock(chain_lock)
def load_from_disk():
    if not os.path.isfile(CHAIN_PATH):
        return
    try:
        with open(CHAIN_PATH, "rb") as f:
            msg_len = int(binascii.hexlify(f.read(4) or b'\x00'), 16) # 前面四个字节是二进制的总字节数，其后是这么多个字节数的内容
            new_blocks = deserialize(f.read(msg_len))  #一次读取完毕，并反序列化
            logger.info(f"loading chain from disk with {len(new_blocks)} blocks")
            for block in new_blocks:  #把每一个区块接上去
                connect_block(block)
    except Exception:
        logger.exception('load chain failed, starting from genesis')


# UTXO set
# ----------------------------------------------------------------------------

utxo_set: Mapping[OutPoint, UnspentTxOut] = {} # OutPoint（UTXO所在的交易ID和检索号）作为键，UnspentTxOut作为值


def add_to_utxo(txout, tx, idx, is_coinbase, height):
    utxo = UnspentTxOut(   
        *txout, # TxOut类型，只包含发送地址和数量
        txid=tx.id, txout_idx=idx, is_coinbase=is_coinbase, height=height)

    logger.info(f'adding tx outpoint {utxo.outpoint} to utxo_set')
    utxo_set[utxo.outpoint] = utxo


def rm_from_utxo(txid, txout_idx): #删除字典元素
    del utxo_set[OutPoint(txid, txout_idx)]


def find_utxo_in_list(txin, txns) -> UnspentTxOut:
    txid, txout_idx = txin.to_spend
    try:
        txout = [t for t in txns if t.id == txid][0].txouts[txout_idx] # [t for t in txns if t.id == txid][0] 是找到的一个交易
    except Exception:
        return None

    return UnspentTxOut(
        *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=txout_idx)


# Proof of work
# ----------------------------------------------------------------------------

def get_next_work_required(prev_block_hash: str) -> int:   #计算下一个区块的难度值
    """
    Based on the chain, return the number of difficulty bits the next block
    must solve.
    """
    if not prev_block_hash: #如果是创世区块，直接返回系统的初始难度值参数
        return Params.INITIAL_DIFFICULTY_BITS

    (prev_block, prev_height, _) = locate_block(prev_block_hash) #获得上一个区块及其高度

    if (prev_height + 1) % Params.DIFFICULTY_PERIOD_IN_BLOCKS != 0:  #如果没到重新设置难度值的时候，则沿用上一个区块的难度值
        return prev_block.bits

      
    #下面是恰好要调整区块难度值的处理代码
    with chain_lock:
        # #realname CalculateNextWorkRequired
        period_start_block = active_chain[max(
            prev_height - (Params.DIFFICULTY_PERIOD_IN_BLOCKS - 1), 0)]  #找到上一个难度的初始区块

    actual_time_taken = prev_block.timestamp - period_start_block.timestamp #计算上一个难度的所有区块的时间总和

    #调整难度值
    if actual_time_taken < Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:  
        # Increase the difficulty
        return prev_block.bits + 1
    elif actual_time_taken > Params.DIFFICULTY_PERIOD_IN_SECS_TARGET:
        return prev_block.bits - 1
    else:
        # Wow, that's unlikely.
        return prev_block.bits


def assemble_and_solve_block(pay_coinbase_to_addr, txns=None):   #生成新区块
    """
    Construct a Block by pulling transactions from the mempool, then mine it.
    """
    with chain_lock:
        prev_block_hash = active_chain[-1].id if active_chain else None

    block = Block(
        version=0,
        prev_block_hash=prev_block_hash,
        merkle_hash='',
        timestamp=int(time.time()),
        bits=get_next_work_required(prev_block_hash),
        nonce=0,
        txns=txns or [],
    )

    if not block.txns:
        block = select_from_mempool(block)  #选择待验证交易，打包进该区块

    fees = calculate_fees(block)  #计算费用
    my_address = init_wallet()[2]  #当前节点的地址
    coinbase_txn = Transaction.create_coinbase(
        my_address, (get_block_subsidy() + fees), len(active_chain))  #构建coinbase交易
    block = block._replace(txns=[coinbase_txn, *block.txns])  # coinbase交易放在第一个位置
    block = block._replace(merkle_hash=get_merkle_root_of_txns(block.txns).val)

    if len(serialize(block)) > Params.MAX_BLOCK_SERIALIZED_SIZE:
        raise ValueError('txns specified create a block too large')

    return mine(block) #挖矿


def calculate_fees(block) -> int: #输入减去输出，就是费用
    """
    Given the txns in a Block, subtract the amount of coin output from the
    inputs. This is kept as a reward by the miner.
    """
    fee = 0

    def utxo_from_block(txin):
        tx = [t.txouts for t in block.txns if t.id == txin.to_spend.txid]
        return tx[0][txin.to_spend.txout_idx] if tx else None

    def find_utxo(txin):
        #to_spend或者取值一个OutPoint实例，要么取值None，后者表示所在交易将是Coinbase交易。
        return utxo_set.get(txin.to_spend) or utxo_from_block(txin) 

    for txn in block.txns:
        spent = sum(find_utxo(i).value for i in txn.txins)
        sent = sum(o.value for o in txn.txouts)
        fee += (spent - sent)

    return fee


def get_block_subsidy() -> int: #返回当前区块奖励数
    halvings = len(active_chain) // Params.HALVE_SUBSIDY_AFTER_BLOCKS_NUM  #减半的次数

    if halvings >= 64: #如果减半次数>=64，新生区块没有奖励
        return 0

    return 50 * Params.LET_PER_COIN // (2 ** halvings)  #当前区块奖励


# Signal to communicate to the mining thread that it should stop mining because
# we've updated the chain with a new block.
mine_interrupt = threading.Event()


def mine(block):
    start = time.time()
    nonce = 0
    target = (1 << (256 - block.bits))
    mine_interrupt.clear()

    while int(sha256d(block.header(nonce)), 16) >= target:
        nonce += 1

        if nonce % 10000 == 0 and mine_interrupt.is_set():  #停止挖矿的条件
            logger.info('[mining] interrupted')
            mine_interrupt.clear()
            return None

    block = block._replace(nonce=nonce)
    duration = int(time.time() - start) or 0.001
    khs = (block.nonce // duration) // 1000
    logger.info(
        f'[mining] block found! {duration} s - {khs} KH/s - {block.id}')

    return block


def mine_forever():
    while True:
        my_address = init_wallet()[2]
        block = assemble_and_solve_block(my_address)

        if block:
            connect_block(block)
            save_to_disk()   #每次挖到一个新块，则重新保存所有区块到block.dat


# Validation
# ----------------------------------------------------------------------------


def validate_txn(txn: Transaction,
                 as_coinbase: bool = False,
                 siblings_in_block: Iterable[Transaction] = None,
                 allow_utxo_from_mempool: bool = True,
                 ) -> Transaction:
    """
    Validate a single transaction. Used in various contexts, so the
    parameters facilitate different uses.
    """
    txn.validate_basics(as_coinbase=as_coinbase)

    available_to_spend = 0

    for i, txin in enumerate(txn.txins):
        utxo = utxo_set.get(txin.to_spend)  #直接从utxo_set里查

        if siblings_in_block:
            utxo = utxo or find_utxo_in_list(txin, siblings_in_block)

        if allow_utxo_from_mempool:
            utxo = utxo or find_utxo_in_mempool(txin)

        if not utxo:
            raise TxnValidationError(
                f'Could find no UTXO for TxIn[{i}] -- orphaning txn',
                to_orphan=txn)

        if utxo.is_coinbase and \
                (get_current_height() - utxo.height) < \
                Params.COINBASE_MATURITY:   #如果是创世交易产生的UTXO，则检查目前能否被花费
            raise TxnValidationError(f'Coinbase UTXO not ready for spend')

        try:
            validate_signature_for_spend(txin, utxo, txn)  #检查签名是否正确
        except TxUnlockError:
            raise TxnValidationError(f'{txin} is not a valid spend of {utxo}')

        available_to_spend += utxo.value

    if available_to_spend < sum(o.value for o in txn.txouts):
        raise TxnValidationError('Spend value is more than available')

    return txn


def validate_signature_for_spend(txin, utxo: UnspentTxOut, txn):
    pubkey_as_addr = pubkey_to_address(txin.unlock_pk)
    verifying_key = ecdsa.VerifyingKey.from_string(
        txin.unlock_pk, curve=ecdsa.SECP256k1)

    if pubkey_as_addr != utxo.to_address:
        raise TxUnlockError("Pubkey doesn't match")

    try:
        spend_msg = build_spend_message(
            txin.to_spend, txin.unlock_pk, txin.sequence, txn.txouts) #注意txn.txouts是该交易的所有输出，说明每个txin都知道交易输出
        verifying_key.verify(txin.unlock_sig, spend_msg) #验证解锁签名unlock_sig是否是公钥unlock_pk对应的私钥对spend_msg的签名
    except Exception:
        logger.exception('Key verification failed')
        raise TxUnlockError("Signature doesn't match")

    return True


def build_spend_message(to_spend, pk, sequence, txouts) -> bytes:
    """This should be ~roughly~ equivalent to SIGHASH_ALL."""
    return sha256d(
        serialize(to_spend) + str(sequence) +
        binascii.hexlify(pk).decode() + serialize(txouts)).encode()


@with_lock(chain_lock)
def validate_block(block: Block) -> Block:
    if not block.txns:
        raise BlockValidationError('txns empty')  #新生成区块必有coinbase交易

    if block.timestamp - time.time() > Params.MAX_FUTURE_BLOCK_TIME:
        raise BlockValidationError('Block timestamp too far in future')  # 该新区块的时间戳不能太超前

    if int(block.id, 16) > (1 << (256 - block.bits)):
        raise BlockValidationError("Block header doesn't satisfy bits")  # 区块ID必须小于指定的难度bits

    if [i for (i, tx) in enumerate(block.txns) if tx.is_coinbase] != [0]:
        raise BlockValidationError('First txn must be coinbase and no more') # 第一个交易必然是coinbase交易

    try:
        for i, txn in enumerate(block.txns):
            txn.validate_basics(as_coinbase=(i == 0))      #对交易的合法性进行初步检查
    except TxnValidationError:
        logger.exception(f"Transaction {txn} in {block} failed to validate")
        raise BlockValidationError('Invalid txn {txn.id}')

    if get_merkle_root_of_txns(block.txns).val != block.merkle_hash:  #查看merkle_hash是否对得上
        raise BlockValidationError('Merkle hash invalid')

    if block.timestamp <= get_median_time_past(11):  #时间戳不能小于前面第五个区块，创世区块不用考虑这一条
        raise BlockValidationError('timestamp too old')

    if not block.prev_block_hash and not active_chain:
        # This is the genesis block.
        prev_block_chain_idx = ACTIVE_CHAIN_IDX
    else:   #创世区块不考虑
        prev_block, prev_block_height, prev_block_chain_idx = locate_block(
            block.prev_block_hash)

        if not prev_block:
            raise BlockValidationError(
                f'prev block {block.prev_block_hash} not found in any chain',
                to_orphan=block)  # 后面的e.to_orphan就是这个block

        # No more validation for a block getting attached to a branch.
        if prev_block_chain_idx != ACTIVE_CHAIN_IDX:
            return block, prev_block_chain_idx

        # Prev. block found in active chain, but isn't tip => new fork.
        elif prev_block != active_chain[-1]:
            return block, prev_block_chain_idx + 1  # Non-existent

    if get_next_work_required(block.prev_block_hash) != block.bits:  #如果当前区块的bits参数不合法的情况
        raise BlockValidationError('bits is incorrect')

    for txn in block.txns[1:]:   #对每个交易的合法性进行检查
        try:
            validate_txn(txn, siblings_in_block=block.txns[1:],
                         allow_utxo_from_mempool=False)
        except TxnValidationError:
            msg = f"{txn} failed to validate"
            logger.exception(msg)
            raise BlockValidationError(msg)

    return block, prev_block_chain_idx


# mempool
# ----------------------------------------------------------------------------

# Set of yet-unmined transactions.
mempool: Dict[str, Transaction] = {}  #全局变量mempool，这是代码里唯一的赋值操作，因此代码中无一处声明其为全局变量。

# Set of orphaned (i.e. has inputs referencing yet non-existent UTXOs)
# transactions.
orphan_txns: Iterable[Transaction] = []  #仅仅识别orphan_txns并添加进该列表，并未进一步处理


def find_utxo_in_mempool(txin) -> UnspentTxOut:
    txid, idx = txin.to_spend

    try:
        txout = mempool[txid].txouts[idx]
    except Exception:
        logger.debug("Couldn't find utxo in mempool for %s", txin)
        return None

    return UnspentTxOut(
        *txout, txid=txid, is_coinbase=False, height=-1, txout_idx=idx)


def select_from_mempool(block: Block) -> Block:
    """Fill a Block with transactions from the mempool."""
    added_to_block = set()

    def check_block_size(b) -> bool:
        return len(serialize(block)) < Params.MAX_BLOCK_SERIALIZED_SIZE

    def try_add_to_block(block, txid) -> Block:
        if txid in added_to_block:
            return block

        tx = mempool[txid]

        # For any txin that can't be found in the main chain, find its
        # transaction in the mempool (if it exists) and add it to the block.
        for txin in tx.txins:
            if txin.to_spend in utxo_set:
                continue   # 如果该交易的所有txin都在utxo_set里，则该交易可以添加进区块

            in_mempool = find_utxo_in_mempool(txin)

            if not in_mempool:
                logger.debug(f"Couldn't find UTXO for {txin}")
                return None

            block = try_add_to_block(block, in_mempool.txid) #如果该交易有txin不在utxo_set里，而是其所在交易未被打包，则先打包其所在的交易
            if not block:
                logger.debug(f"Couldn't add parent")
                return None

        newblock = block._replace(txns=[*block.txns, tx])  #添加该交易

        if check_block_size(newblock):
            logger.debug(f'added tx {tx.id} to block')
            added_to_block.add(txid)
            return newblock
        else:
            return block

    for txid in mempool:   #对mempool里的所有交易，依次检查，并试图添加进区块
        newblock = try_add_to_block(block, txid)

        if check_block_size(newblock):
            block = newblock
        else:
            break

    return block


def add_txn_to_mempool(txn: Transaction):
    if txn.id in mempool:
        logger.info(f'txn {txn.id} already seen')
        return

    try:
        txn = validate_txn(txn)
    except TxnValidationError as e:
        if e.to_orphan:
            logger.info(f'txn {e.to_orphan.id} submitted as orphan')
            orphan_txns.append(e.to_orphan)
        else:
            logger.exception(f'txn rejected')
    else:
        logger.info(f'txn {txn.id} added to mempool')
        mempool[txn.id] = txn

        for peer in peer_hostnames:
            send_to_peer(txn, peer)


# Merkle trees
# ----------------------------------------------------------------------------

class MerkleNode(NamedTuple): #一个节点，及其两个孩子节点
    val: str
    children: Iterable = None


def get_merkle_root_of_txns(txns):
    return get_merkle_root(*[t.id for t in txns])


@lru_cache(maxsize=1024)
def get_merkle_root(*leaves: Tuple[str]) -> MerkleNode:
    """Builds a Merkle tree and returns the root given some leaf values."""
    if len(leaves) % 2 == 1:
        leaves = leaves + (leaves[-1],) #确保叶子节点由偶数个

    def find_root(nodes): # 递归调用，计算root
        newlevel = [
            MerkleNode(sha256d(i1.val + i2.val), children=[i1, i2])
            for [i1, i2] in _chunks(nodes, 2)
        ]

        return find_root(newlevel) if len(newlevel) > 1 else newlevel[0]

    return find_root([MerkleNode(sha256d(l)) for l in leaves])


# Peer-to-peer
# ----------------------------------------------------------------------------

peer_hostnames = {p for p in os.environ.get('TC_PEERS', '').split(',') if p} # 从环境变量TC_PEERS读取节点，存储在peer_hostnames里， 不然为空

# Signal when the initial block download has completed.
ibd_done = threading.Event()


class GetBlocksMsg(NamedTuple):  # Request blocks during initial sync
    """
    See https://bitcoin.org/en/developer-guide#blocks-first
    """
    from_blockid: str

    CHUNK_SIZE = 50

    def handle(self, sock, peer_hostname):
        logger.debug(f"[p2p] recv getblocks from {peer_hostname}")

        _, height, _ = locate_block(self.from_blockid, active_chain)

        # If we don't recognize the requested hash as part of the active
        # chain, start at the genesis block.
        height = height or 1

        with chain_lock:
            blocks = active_chain[height:(height + self.CHUNK_SIZE)]

        logger.debug(f"[p2p] sending {len(blocks)} to {peer_hostname}")
        send_to_peer(InvMsg(blocks), peer_hostname)


class InvMsg(NamedTuple):  # Convey blocks to a peer who is doing initial sync
    blocks: Iterable[str]

    def handle(self, sock, peer_hostname):
        logger.info(f"[p2p] recv inv from {peer_hostname}")

        new_blocks = [b for b in self.blocks if not locate_block(b.id)[0]]

        if not new_blocks: #如果不能收到新区块，说明已经同步到最新区块
            logger.info('[p2p] initial block download complete')
            ibd_done.set()
            return

        for block in new_blocks: #添加收到的多个区块
            connect_block(block)

        new_tip_id = active_chain[-1].id  
        logger.info(f'[p2p] continuing initial block download at {new_tip_id}')

        with chain_lock:
            # "Recursive" call to continue the initial block sync.
            send_to_peer(GetBlocksMsg(new_tip_id))   # 再次调用同步区块函数，从new_tip_id开始，直到同步到最新区块


class GetUTXOsMsg(NamedTuple):  # List all UTXOs
    def handle(self, sock, peer_hostname):
        sock.sendall(encode_socket_data(list(utxo_set.items())))


class GetMempoolMsg(NamedTuple):  # List the mempool
    def handle(self, sock, peer_hostname):
        sock.sendall(encode_socket_data(list(mempool.keys())))


class GetActiveChainMsg(NamedTuple):  # Get the active chain in its entirety.
    def handle(self, sock, peer_hostname):
        sock.sendall(encode_socket_data(list(active_chain)))


class AddPeerMsg(NamedTuple): 
    peer_hostname: str

    def handle(self, sock, peer_hostname):
        peer_hostnames.add(self.peer_hostname)  #全局变量peer_hostnames添加元素操作，指针未变，因此不需声明全局变量


def read_all_from_socket(req) -> object:  #读取socket数据，并将其反序列化
    data = b''
    # Our protocol is: first 4 bytes signify msg length.
    msg_len = int(binascii.hexlify(req.recv(4) or b'\x00'), 16)

    while msg_len > 0:
        tdat = req.recv(1024)
        data += tdat
        msg_len -= len(tdat)

    return deserialize(data.decode()) if data else None


def send_to_peer(data, peer=None): #向指定的peer或者随机指定的一个peer发送data数据
    """Send a message to a (by default) random peer."""
    global peer_hostnames

    peer = peer or random.choice(list(peer_hostnames))  #发送给指定的peer，或者随机指定一个peer
    tries_left = 3

    while tries_left > 0:
        try:
            with socket.create_connection((peer, PORT), timeout=1) as s:
                s.sendall(encode_socket_data(data))
        except Exception:
            logger.exception(f'failed to send to peer {peer}')
            tries_left -= 1
            time.sleep(2)
        else:
            return

    logger.info(f"[p2p] removing dead peer {peer}")
    
    # 对全局变量进行赋值，指针变化，因此该函数需要对peer_hostnames进行声明
    peer_hostnames = {x for x in peer_hostnames if x != peer} #超过三次不能建立TCP连接，则剔除该节点
    

def int_to_8bytes(a: int) -> bytes: return binascii.unhexlify(f"{a:0{8}x}")


def encode_socket_data(data: object) -> bytes:
    """Our protocol is: first 4 bytes signify msg length."""
    to_send = serialize(data).encode()
    return int_to_8bytes(len(to_send)) + to_send   #前四个字节是二进制数据，表示要传输的数据to_send的长度，其后才是to_send


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class TCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = read_all_from_socket(self.request)  #右侧函数read_all_from_socket负责读取socket数据并对其反序列化，返回一个实例
        peer_hostname = self.request.getpeername()[0]   # request.getpeername()返回一个tuple('127.0.0.1', 51602)，包含IP和PORT
        peer_hostnames.add(peer_hostname)  #添加新节点；添加元素不改变指针，无需声明全局变量

        if hasattr(data, 'handle') and isinstance(data.handle, Callable):
            logger.info(f'received msg {data} from peer {peer_hostname}')
            data.handle(self.request, peer_hostname)
        elif isinstance(data, Transaction):  #收到一个交易实例
            logger.info(f"received txn {data.id} from peer {peer_hostname}")
            add_txn_to_mempool(data)
        elif isinstance(data, Block): #收到一个区块实例
            logger.info(f"received block {data.id} from peer {peer_hostname}")
            connect_block(data)


# Wallet
# ----------------------------------------------------------------------------

WALLET_PATH = os.environ.get('TC_WALLET_PATH', 'wallet.dat')  # 获取环境变量TC_WALLET_PATH，如果该变量不存在，则返回wallet.dat


def pubkey_to_address(pubkey: bytes) -> str:  #由公钥生成地址
    if 'ripemd160' not in hashlib.algorithms_available:
        raise RuntimeError('missing ripemd160 hash algorithm')

    sha = hashlib.sha256(pubkey).digest()
    ripe = hashlib.new('ripemd160', sha).digest()
    return b58encode_check(b'\x00' + ripe)


@lru_cache()
def init_wallet(path=None):
    path = path or WALLET_PATH

    if os.path.exists(path):
        with open(path, 'rb') as f:   
            signing_key = ecdsa.SigningKey.from_string(
                f.read(), curve=ecdsa.SECP256k1)
    else:
        logger.info(f"generating new wallet: '{path}'")
        signing_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        with open(path, 'wb') as f:
            f.write(signing_key.to_string()) #钱包里仅存储了私钥

    verifying_key = signing_key.get_verifying_key()
    my_address = pubkey_to_address(verifying_key.to_string())  #这两行代码根据私钥来计算公钥
    logger.info(f"your address is {my_address}")

    return signing_key, verifying_key, my_address  #init_wallet初始化钱包，返回私钥，公钥，和地址


# Misc. utilities
# ----------------------------------------------------------------------------

class BaseException(Exception):
    def __init__(self, msg):
        self.msg = msg


class TxUnlockError(BaseException):
    pass


class TxnValidationError(BaseException):
    def __init__(self, *args, to_orphan: Transaction = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_orphan = to_orphan


class BlockValidationError(BaseException):
    def __init__(self, *args, to_orphan: Block = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_orphan = to_orphan


def serialize(obj) -> str:  #实例转化成字符串
    """NamedTuple-flavored serialization to JSON."""
    def contents_to_primitive(o):
        if hasattr(o, '_asdict'): #如果该实例有将自身转换成字典的方法，就执行该方法，返回字典
            o = {**o._asdict(), '_type': type(o).__name__}
        elif isinstance(o, (list, tuple)): #如果是list或者tuple实例，则返回元素列表
            return [contents_to_primitive(i) for i in o]
        elif isinstance(o, bytes): #如果是二进制，则转换成16进制
            return binascii.hexlify(o).decode()
        elif not isinstance(o, (dict, bytes, str, int, type(None))): 
            raise ValueError(f"Can't serialize {o}")

        if isinstance(o, Mapping):  #映射类型，即字典
            for k, v in o.items():
                o[k] = contents_to_primitive(v)

        return o

    return json.dumps(
        contents_to_primitive(obj), sort_keys=True, separators=(',', ':'))


def deserialize(serialized: str) -> object: #将json字符串恢复成对象
    """NamedTuple-flavored serialization from JSON."""
    gs = globals()

    def contents_to_objs(o):
        if isinstance(o, list):
            return [contents_to_objs(i) for i in o]
        elif not isinstance(o, Mapping):
            return o

        _type = gs[o.pop('_type', None)]
        bytes_keys = {
            k for k, v in get_type_hints(_type).items() if v == bytes}

        for k, v in o.items():
            o[k] = contents_to_objs(v)

            if k in bytes_keys:
                o[k] = binascii.unhexlify(o[k]) if o[k] else o[k]

        return _type(**o)

    return contents_to_objs(json.loads(serialized))


def sha256d(s: Union[str, bytes]) -> str:
    """A double SHA-256 hash."""
    if not isinstance(s, bytes):
        s = s.encode()  #转换成二进制字节码

    return hashlib.sha256(hashlib.sha256(s).digest()).hexdigest() #hashlib.sha256()返回一个sha256对象，调用.hexdigest方法返回16进制字符串


def _chunks(l, n) -> Iterable[Iterable]: #把l变成长度为n的多个段，最后一个段的长度小于等于n。
    return (l[i:i + n] for i in range(0, len(l), n))


# Main
# ----------------------------------------------------------------------------

PORT = os.environ.get('TC_PORT', 9999) #如果没有指定端口，就使用默认端口9999


def main():
    load_from_disk()

    workers = []
    server = ThreadedTCPServer(('0.0.0.0', PORT), TCPHandler)

    def start_worker(fnc):
        workers.append(threading.Thread(target=fnc, daemon=True))  #添加线程并将其启动
        workers[-1].start()

    logger.info(f'[p2p] listening on {PORT}')
    start_worker(server.serve_forever)  #启动监听端口的线程

    if peer_hostnames:
        logger.info(
            f'start initial block download from {len(peer_hostnames)} peers')
        # send_to_peer(data, peer=None): #向指定的peer或者随机指定的一个peer发送data数据                                                         
        send_to_peer(GetBlocksMsg(active_chain[-1].id))   #把初始化完毕的实例GetBlocksMsg(active_chain[-1].id)，作为data参数传送出去
        
        # ibd （initial block download）的缩写，
           #是threading.Event()类型，调用该方法的线程会被阻塞，如果设置了timeout参数，超时后，线程会停止阻塞继续执行；                                                         
        ibd_done.wait(60.)  # Wait a maximum of 60 seconds for IBD to complete. 
    start_worker(mine_forever)  #启动挖矿线程
    [w.join() for w in workers]


if __name__ == '__main__':
    signing_key, verifying_key, my_address = init_wallet()
    main()
