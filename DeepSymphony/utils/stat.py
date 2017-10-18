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
