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

from p2p.Peer import Peer
from ds.Block import Block


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

		return json.dumps(
	contents_to_primitive(obj), sort_keys=True, separators=(',', ':'))


    @classmethod
    def deserialize(cls, serialized: str) -> object:
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

    @classmethod
    def sha256d(cls, s: Union[str, bytes]) -> str:
		"""A double SHA-256 hash."""
		if not isinstance(s, bytes):
			s = s.encode()

		return hashlib.sha256(hashlib.sha256(s).digest()).hexdigest()

    @classmethod
    def encode_chain_data(cls, chain: Iterable[Block]) -> bytes:
		"""Our protocol is: first 4 bytes signify msg length."""
		def int_to_8bytes(a: int) -> bytes:
			return binascii.unhexlify(f"{a:0{8}x}")
		block_len = len(chain)
		to_send = cls.serialize(chain).encode()
		msg_len = len(to_send)
		return int_to_8bytes(block_len) + int_to_8bytes(msg_len) + to_send
	




    @classmethod
    def send_to_peer(cls, data, peer)->bool:
		#peer = peer or random.choice(list(peers))
		#if not isinstance(peer, Peer):
		#	logger.error(f"{peer} is not instance of Peer class" )
		#	return False
		tries_left = 3
	
		while tries_left > 0:
			try:
				with socket.create_connection(*peer(), timeout=1) as s:
					s.sendall(cls.encode_socket_data(data))
				return True
			except Exception:
				logger.exception(f'failed to send to peer {peer}')
				tries_left -= 1
				time.sleep(2)
			else:
				return False

    @classmethod
    def with_lock(cls, lock: RLock):
		def dec(func):
			@wraps(func)
			def wrapper(*args, **kwargs):
				with lock:
					return func(*args, **kwargs)
			return wrapper
		return dec
