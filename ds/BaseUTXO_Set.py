class BaseUTXO_Set(object):

    def __init__(self):
        self.utxoSet = {}
    def get(self):
        pass

    def add_to_utxo(self, txout, tx, idx, is_coinbase, height):
        pass

    def rm_from_utxo(self, txid, txout_idx):
        pass

    def find_utxo_in_list(cls, txin, txns):
       pass
