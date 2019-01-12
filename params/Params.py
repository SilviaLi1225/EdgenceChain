
from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
from ds.Block  import (OutPoint, TxIn, TxOut, UnspentTxOut, Transaction,
                       Block)
from p2p.Peer import Peer

class Params:
    # The infamous max block size.
    MAX_BLOCK_SERIALIZED_SIZE = 1000000  # bytes = 1MB

    # Coinbase transaction outputs can be spent after this many blocks have
    # elapsed since being mined.
    #
    # This is "100" in bitcoin core.
    COINBASE_MATURITY = 2

    # Accept blocks timestamped as being from the future, up to this amount.
    MAX_FUTURE_BLOCK_TIME = (60 * 60 * 2)

    # The number of LET per coin. #realname COIN
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


    genesis_block = Block(
        version=0,
        prev_block_hash=None,
        merkle_hash=(
            '7118894203235a955a908c0abfc6d8fe6edec47b0a04ce1bf7263da3b4366d22'),
        timestamp=1501821412,
        bits=24,
        nonce=10126761,
        txns=[Transaction(
                txins=[TxIn(
                    to_spend=None, unlock_sig=b'0', unlock_pk=None, sequence=0)],
                txouts=[TxOut(
                    value=5000000000,
                    to_address='143UVyz7ooiAv1pMqbwPPpnH4BV9ifJGFF')],
                locktime=None)]
        )
    # list of peers
    PEERS_FILE =  'peers.conf'
    PEERS: Iterable[Peer] = list([Peer('127.0.0.1', 9999),
                      Peer('127.0.0.1', 9998),
                      Peer('localhost', 9999)])

