class BaseBlockChain(object):


    def __init__(self, idx: int=0, chain = []):
        self.index = idx
        self.chain = chain

    @property
    def height(self):
        pass

    @property
    def idx(self):
        pass

    def find_txout_for_txin(self, txin):
        pass

    def disconnect_block(self, mempool, utxo_set):
        pass

    def connect_block(self, block, active_chain, side_branches, mempool, utxo_set, mine_interrupt,\
                      peers, doing_reorg):
        pass
