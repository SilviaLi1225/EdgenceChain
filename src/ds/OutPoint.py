from typing import NamedTuple

OutPoint = NamedTuple('OutPoint', [('txid', str), ('txout_idx', int)])
