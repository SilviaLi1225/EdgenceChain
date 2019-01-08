import logging
import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from edgencechain.chain import (GetBlocksMsg, TCPHandler, ThreadedTCPServer,
                                active_chain, ibd_done, init_wallet,
                                load_from_disk, mine_forever, peer_hostnames,
                                send_to_peer)

PORT = os.environ.get('TC_PORT', 9999)

logging.basicConfig(
    level=getattr(logging, os.environ.get('TC_LOG_LEVEL', 'INFO')),
    format='[%(asctime)s][%(module)s:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    load_from_disk()

    workers = []
    server = ThreadedTCPServer(('0.0.0.0', PORT), TCPHandler)

    def start_worker(fnc):
        workers.append(threading.Thread(target=fnc, daemon=True))
        workers[-1].start()

    logger.info(f'[p2p] listening on {PORT}')
    start_worker(server.serve_forever)

    if peer_hostnames:
        logger.info(
            f'start initial block download from {len(peer_hostnames)} peers')
        send_to_peer(GetBlocksMsg(active_chain[-1].id))
        ibd_done.wait(60.)  # Wait a maximum of 60 seconds for IBD to complete.

    start_worker(mine_forever)
    [w.join() for w in workers]


if __name__ == '__main__':
    signing_key, verifying_key, my_address = init_wallet()
    main()
