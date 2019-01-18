from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
import os,logging,binascii


logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
"""
>>> from Peer import Peer
>>> peer = Peer()
>>> peer.get()
('localhost', 9999)
>>> peer = Peer('10.108.01.13')
>>> peer.get()
('10.108.01.13', 9999)
>>> peer = Peer('10.108.01.13',18)
>>> peer.get()
('10.108.01.13', 18)
"""
from params import Params
from utils.Utils import Utils

class Peer(NamedTuple):
    ip: str = 'localhost'
    port: int = 9999

    def __call__(self):
        return str(self.ip), int(self.port)

    @property
    def id(self):
        return Utils.sha256d(Utils.serialize(self))

    @classmethod
    def init_peers(cls, peerfile = Params.Params.PEERS_FILE)->Iterable[NamedTuple]:
        if not os.path.exists(peerfile):
            peers: Iterable[Peer] =[]
            for peer in Params.Params.PEERS:
                peers.append(Peer(*peer))
            try:
                with open(peerfile, "wb") as f:
                    logger.info(f"saving {len(peers)} hostnames")
                    f.write(Utils.encode_socket_data(list(peers)))
            except Exception:
                logger.exception('saving peers exception')
        else:
            try:
                with open(peerfile, "rb") as f:
                    msg_len = int(binascii.hexlify(f.read(4) or b'\x00'), 16)
                    gs = dict()
                    gs['Peer'] = globals()['Peer']
                    peers = Utils.deserialize(f.read(msg_len), gs)
                    logger.info(f"loading peers with {len(peers)} hostnames")
            except Exception:
                logger.exception('loading peers exception')
                peers = []
        return peers


