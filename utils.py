class Config(object):
    # kwargs is a DICTIONARY
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)