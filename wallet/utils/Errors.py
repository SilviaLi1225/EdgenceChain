

class BaseException(Exception):
    def __init__(self, msg):
        self.msg = msg

class TxUnlockError(BaseException):
    pass

class ChainFileLostError (BaseException):
    pass

class TxnValidationError(BaseException):
    def __init__(self, *args, to_orphan = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_orphan = to_orphan

class BlockValidationError(BaseException):
    def __init__(self, *args, to_orphan=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_orphan = to_orphan


