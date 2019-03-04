from typing import (
    Iterable, NamedTuple, Dict, Mapping, Union, get_type_hints, Tuple,
    Callable)
import os,logging,binascii
#use regular expression to match non-IP
import re
import socket

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
from params.Params import Params
from utils.Utils import Utils

class Peer(NamedTuple):
    ip: str = 'localhost'
    port: int = 9999

    def __call__(self):
        return str(self.ip), int(self.port)

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.ip == other.ip and self.port == other.port

    def  __hash__(self):
        return hash(f'{self.ip}{self.port}')

    @property
    def id(self):
        return Utils.sha256d(Utils.serialize(self))

    @classmethod
    def init_peers(cls, peerfile = Params.PEERS_FILE)->Iterable[NamedTuple]:
        if not os.path.exists(peerfile):
            peers: Iterable[Peer] =[]
            for peer in Params.PEERS:
                if (str(peer[0]) == '127.0.0.1' and int(peer[1]) == Params.PORT_CURRENT) or \
                    (str(peer[0]) == 'localhost' and int(peer[1]) == Params.PORT_CURRENT):
                    pass
                else:
                    #match IP address
                    if re.match(r'(?<![\.\d])(?:25[0-5]\.|2[0-4]\d\.|[01]?\d\d?\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\.\d])',peer[0])==None:
                        #case I,is not a IP address use DNS
                        #replace the peer's name to the resolved name(which should be a IP addr.)
                        #catch errors if the name is not valid
                        try:
                            peer[0]=socket.gethostbyname(peer[0])
                        except Exception:
                            logger.exception(f"[p2p] {peer[0]} can not be resolved , maybe not a valid name")
                    else:
                        pass
                    #append the IP to the peers
                    peers.append(Peer(str(peer[0]), int(peer[1])))
            try:
                with open(peerfile, "wb") as f:
                    logger.info(f"[p2p] saving {len(peers)} hostnames")
                    f.write(Utils.encode_socket_data(list(peers)))
            except Exception:
                logger.exception(f'[p2p] saving peers exception')
                return []
        else:
            try:
                with open(peerfile, "rb") as f:
                    msg_len = int(binascii.hexlify(f.read(4) or b'\x00'), 16)
                    gs = dict()
                    gs['Peer'] = globals()['Peer']
                    peers = Utils.deserialize(f.read(msg_len), gs)
                    peers = list(set(peers))
                    logger.info(f"[p2p] loading peers with {len(peers)} hostnames")
            except Exception:
                logger.exception(f'[p2p] loading peers exception')
                peers = []
        return peers

    @classmethod
    def save_peers(cls, peers: Iterable[NamedTuple], peerfile = Params.PEERS_FILE):
        try:
            with open(peerfile, "wb") as f:
                logger.info(f"[p2p] saving {len(peers)} hostnames")
                f.write(Utils.encode_socket_data(list(peers)))
        except Exception:
            logger.exception('[p2p] saving peers exception')
