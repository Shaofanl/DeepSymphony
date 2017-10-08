import abc


class CoderBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        raise NotImplemented

    @abc.abstractmethod
    def encode(self, seq, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def decode(self, seq, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def event_to_code(self, event):
        raise NotImplemented
