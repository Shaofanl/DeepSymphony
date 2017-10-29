"""
    Longest Common Subsequence
"""
import numpy as np


def eval_lcs(generated, truths, return_seq=False):
    matches = map(lambda t: _LCS(generated, t, return_seq=return_seq),
                  truths)
    return matches


def _LCS(P, Q, return_seq=False):
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
