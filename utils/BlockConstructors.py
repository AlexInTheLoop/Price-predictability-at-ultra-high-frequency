import numpy as np


def non_overlapping_blocks(symbols, block_size):
    n = len(symbols)
    n_blocks = n // block_size
    trimmed_symbols = symbols[:n_blocks * block_size]
    blocks = np.array(trimmed_symbols).reshape(-1, block_size)
    return blocks

def overlapping_blocks(symbols, block_size):

    n = symbols.shape[0]
    if n < block_size:
        return np.empty((0, block_size), dtype=symbols.dtype)

    shape = (n - block_size + 1, block_size)
    strides = (symbols.strides[0], symbols.strides[0])
    return np.lib.stride_tricks.as_strided(symbols, shape=shape, strides=strides)


def cross_overlapping_blocks(context_symbols, target_symbols, k):
    """
    Version optimisée numpy.
    Construit les blocs contextuels + target en O(n).
    """
    n_ctx = len(context_symbols)
    n_tgt = len(target_symbols)
    N = min(n_ctx, n_tgt)

    if N <= k:
        return np.empty((0, k + 1), dtype=context_symbols.dtype)

    # construire blocs context en numpy strides
    shape = (N - k, k)
    strides = (context_symbols.strides[0], context_symbols.strides[0])
    context_blocks = np.lib.stride_tricks.as_strided(context_symbols, shape=shape, strides=strides)

    # target symbols décalés de k
    target_symbols_shifted = target_symbols[k:N]

    # concaténer : output = [context_block_1 + target_1], ...
    return np.hstack([context_blocks, target_symbols_shifted.reshape(-1, 1)])

def cross_non_overlapping_blocks(ctx, tgt, k):
    N = min(len(ctx), len(tgt))
    n_blocks = N // k
    rows = []
    for i in range(n_blocks - 1):
        start = i * k
        block_ctx = tuple(ctx[start : start + k])
        sym_tgt  = tgt[(i + 1) * k]
        rows.append(block_ctx + (sym_tgt,))
    return np.array(rows, dtype=int)

