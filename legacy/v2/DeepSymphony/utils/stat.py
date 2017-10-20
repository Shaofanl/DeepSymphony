import numpy as np


def histogram(items, keys):
    count = np.zeros((len(keys),))
    for ind, key in enumerate(keys):
        for item in items:
            if item.find(key) != -1:
                count[ind] += 1
    return count
#   return np.array([items.count(key) for key in keys])


def histogram_onehot(notes, tokenize, keys):
    if notes.ndim == 2:
        events = [tokenize(note) for note in notes.argmax(1)]
    elif notes.ndim == 1:
        events = [tokenize(note) for note in notes]
    hist = histogram(events, keys)
    hist /= hist.sum()
    return hist


def min_norm(h):
    if h.ndim == 2:
        h -= h.min(1)
        h /= h.sum(1)[:, None]
    elif h.ndim == 1:
        h -= h.min()
        h /= h.sum()
    return h


def norm(h, temperature=1.0):
    if h.ndim == 2:
        h = np.exp(h/temperature)
        h /= h.sum(1)[:, None]
    elif h.ndim == 1:
        h = np.exp(h/temperature)
        h /= h.sum()
    return h


def LCS(P, Q, return_seq=False):
    f = [[0 for j in xrange(len(Q)+1)] for i in xrange(len(P)+1)]

    for i, x in enumerate(P):
        for j, y in enumerate(Q):
            if x == y:
                f[i+1][j+1] = f[i][j]+1
            else:
                f[i+1][j+1] = max(f[i+1][j], f[i][j+1])

    if not return_seq:
        return f[len(P)][len(Q)]

    result = ""
    P_indicator = np.array([False for i in xrange(len(P)+1)], dtype='float')
    Q_indicator = np.array([False for i in xrange(len(Q)+1)], dtype='float')
    x, y = len(P), len(Q)
    while x != 0 and y != 0:
        if f[x][y] == f[x-1][y]:
            x -= 1
        elif f[x][y] == f[x][y-1]:
            y -= 1
        else:
            assert P[x-1] == Q[y-1]
            result = '<{}>'.format(P[x-1]) + result
            x -= 1
            y -= 1
            P_indicator[x-1] = True
            Q_indicator[y-1] = True
    return f[len(P)][len(Q)], result, P_indicator, Q_indicator
