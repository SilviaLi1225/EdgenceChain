
class BaseMemPool(object):
    def __init__(self):
        self.mempool = {}
        self.orphan_txns = []

    def get(self):
        pass

    def find_utxo_in_mempool(self, txin):
        pass

    def select_from_mempool(self, block, utxo_set):
        pass

    def add_txn_to_mempool(self, txn, utxo_set):
        pass
