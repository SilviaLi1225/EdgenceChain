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
from _thread import RLock

from params.Params import Params

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



class Utils(object):

    @classmethod
    def serialize(cls, obj) -> str:
        """NamedTuple-flavored serialization to JSON."""
        def contents_to_primitive(o):
            if hasattr(o, '_asdict'):
                o = {**o._asdict(), '_type': type(o).__name__}
            elif isinstance(o, (list, tuple)):
                return [contents_to_primitive(i) for i in o]
            elif isinstance(o, bytes):
                return binascii.hexlify(o).decode()
            elif not isinstance(o, (dict, bytes, str, int, type(None))):
                raise ValueError(f"Can't serialize {o}")
            if isinstance(o, Mapping):
                for k, v in o.items():
                    o[k] = contents_to_primitive(v)
            return o
        return json.dumps(contents_to_primitive(obj), sort_keys=True, separators=(',', ':'))

    @classmethod
    def deserialize(cls, serialized: str, gs: dict) -> object:
        """NamedTuple-flavored serialization from JSON."""

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

    @classmethod
    def sha256d(cls, s: Union[str, bytes]) -> str:
        """A double SHA-256 hash."""
        if not isinstance(s, bytes):
            s = s.encode()

        return hashlib.sha256(hashlib.sha256(s).digest()).hexdigest()

    @classmethod
    def encode_chain_data(cls, chain: Iterable[NamedTuple]) -> bytes:
        """Our protocol is: first 4 bytes signify msg length."""
        def int_to_8bytes(a: int) -> bytes:
            return binascii.unhexlify(f"{a:0{8}x}")
        block_len = len(chain)
        to_send = cls.serialize(chain).encode()
        msg_len = len(to_send)
        return int_to_8bytes(block_len) + int_to_8bytes(msg_len) + to_send

    @classmethod
    def encode_socket_data(cls, data: object) -> bytes:
        """Our protocol is: first 4 bytes signify msg length."""
        def int_to_8bytes(a: int) -> bytes:
            return binascii.unhexlify(f"{a:0{8}x}")
        to_send = Utils.serialize(data).encode()
        return int_to_8bytes(len(to_send)) + to_send

    @classmethod
    def send_to_peer(cls, data, peer)->bool:
        tries_left = Params.TRIES_MAXIMUM

        while tries_left > 0:
            try:
                with socket.create_connection(peer(), timeout=1) as s:
                    s.sendall(cls.encode_socket_data(data))
            except Exception:
                logger.exception(f'failed to send to {peer}  in {Params.TRIES_MAXIMUM+1-tries_left}th time')
                tries_left -= 1
                time.sleep(2)
                if tries_left <= 0:
                    return False
            else:
                logger.info(f'succeed in sending to {peer} in {Params.TRIES_MAXIMUM+1-tries_left}th time')
                return True


    @classmethod
    def read_all_from_socket(cls, req, gs) -> object:
        data = b''
        # Our protocol is: first 4 bytes signify msg length.
        msg_len = int(binascii.hexlify(req.recv(4) or b'\x00'), 16)

        while msg_len > 0:
            tdat = req.recv(1024)
            data += tdat
            msg_len -= len(tdat)

        return cls.deserialize(data.decode(), gs) if data else None



    @classmethod
    def with_lock(cls, lock: RLock):
        def dec(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with lock:
                    return func(*args, **kwargs)
            return wrapper
        return dec
