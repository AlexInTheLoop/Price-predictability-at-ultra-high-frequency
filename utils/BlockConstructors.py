import numpy as np


def non_overlapping_blocks(symbols, block_size):
    n = len(symbols)
    n_blocks = n // block_size
    trimmed_symbols = symbols[:n_blocks * block_size]
    blocks = np.array(trimmed_symbols).reshape(-1, block_size)
    return blocks

def overlapping_blocks(symbols, block_size):
    n = len(symbols)
    blocks = np.array([],dtype=int)
    for i in range(n - block_size + 1):
        blocks = np.append(blocks, symbols[i:i + block_size])
    blocks = blocks.reshape(-1, block_size)
    return blocks

def cross_overlapping_blocks(context_symbols,target_symbols,k):

    # Determine the effective length to avoid index errors
    N = min(len(context_symbols), len(target_symbols))
    
    rows = []
    for i in range(N - k):
        block_context = tuple(context_symbols[i : i + k])
        symbol_target = target_symbols[i + k]
        rows.append(block_context + (symbol_target,))
    
    return np.array(rows, dtype=int)

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

