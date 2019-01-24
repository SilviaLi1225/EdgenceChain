from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)


class TxOut(NamedTuple):
    """Outputs from a Transaction."""
    # The number of LET this awards.
    value: int

    # The public key of the owner of this Txn.
    to_address: str
