from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)


from utils.Errors import (BaseException, TxUnlockError, TxnValidationError, BlockValidationError)

from utils.Utils import Utils
from params.Params import Params
from wallet.Wallet import Wallet
from ds.UnspentTxOut import UnspentTxOut


import binascii,ecdsa,logging,os
from functools import lru_cache, wraps



logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class MerkleNode(NamedTuple):
    val: str
    children: Iterable = None

    @classmethod
    def get_merkle_root_of_txns(cls, txns):


        @lru_cache(maxsize=1024)
        def get_merkle_root(*leaves: Tuple[str]) -> MerkleNode:
            """Builds a Merkle tree and returns the root given some leaf values."""
            if len(leaves) % 2 == 1:
                leaves = leaves + (leaves[-1],)

            def find_root(nodes):
                def _chunks(l, n) -> Iterable[Iterable]:
                    return (l[i:i + n] for i in range(0, len(l), n))
                newlevel = [
                    MerkleNode(Utils.sha256d(i1.val + i2.val), children=[i1, i2])
                    for [i1, i2] in _chunks(nodes, 2)
                ]

                return find_root(newlevel) if len(newlevel) > 1 else newlevel[0]

            return find_root([MerkleNode(Utils.sha256d(l)) for l in leaves])


        return get_merkle_root(*[t.id for t in txns])


