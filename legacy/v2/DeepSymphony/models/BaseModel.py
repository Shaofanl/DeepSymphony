import abc


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.build(**kwargs)

    @abc.abstractmethod
    def build(self, **kwargs):
        pass

    # override recommended
    def train(self, data_generator):
        self.model.fit_generator(data_generator)

    @abc.abstractmethod
    def generate(self, **kwargs):
        pass
