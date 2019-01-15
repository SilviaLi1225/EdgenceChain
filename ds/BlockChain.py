from ds.Block import Block

from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)

class BlockChain(object):

    def __init__(self, idx=0, chain=[]):
        self.idx = idx
        self.chain: Iterable[Block] = chain



