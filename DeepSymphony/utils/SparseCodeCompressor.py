import numpy as np


class SparseCodeCompressor(object):
    def __init__(self, seq):
        self.compact2sparse = np.unique(seq)
        self.sparse2compact = np.array([None]*(np.max(seq)+1))
        for i, ele in enumerate(self.compact2sparse):
            self.sparse2compact[ele] = i
        self.count = len(self.compact2sparse)

    def to_sparse(self, seq):
        return self.compact2sparse[seq]

    def to_compact(self, seq):
        return self.sparse2compact[seq].astype('int')


if __name__ == '__main__':
    a = [4, 5, 5, 5, 6, 7]
    c = SparseCodeCompressor(a)
    print a
    print c.to_compact(a)
    print c.to_sparse(c.to_compact(a))
