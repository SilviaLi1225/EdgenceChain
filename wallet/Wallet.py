import hashlib
import logging
import os
from functools import lru_cache
import ecdsa
from base58 import b58encode_check



logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



class Wallet(object):
	"""
	wallet = Wallet.init_wallet('wallet.dat')  # return an object using the path
	wallet2 = Wallet(wallet())   #wallet() returns three parameters of a wallet
	"""

	def __init__(self, signing_key, verifying_key, my_address):

		self.signing_key = signing_key
		self.verifying_key  = verifying_key
		self.my_address = my_address

	def __call__(self):
		return self.signing_key, self.verifying_key, self.my_address

	@classmethod
	def pubkey_to_address(cls, pubkey: bytes) -> str:
		if 'ripemd160' not in hashlib.algorithms_available:
			raise RuntimeError('missing ripemd160 hash algorithm')

		sha = hashlib.sha256(pubkey).digest()
		ripe = hashlib.new('ripemd160', sha).digest()
		return str(b58encode_check(b'\x00' + ripe))



	@classmethod
	@lru_cache()
	def init_wallet(cls, path='wallet.dat'):
		if os.path.exists(path):
			with open(path, 'rb') as f:
				signing_key = ecdsa.SigningKey.from_string(
					f.read(), curve=ecdsa.SECP256k1)
		else:
			logger.info(f"[wallet] generating new wallet: '{path}'")
			signing_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
			with open(path, 'wb') as f:
				f.write(signing_key.to_string())

		verifying_key = signing_key.get_verifying_key()
		my_address = Wallet.pubkey_to_address(verifying_key.to_string())
		logger.info(f"[wallet] your address is {my_address}")

		return cls(signing_key, verifying_key, my_address)
