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