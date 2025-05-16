import pandas as pd
import numpy as np
import itertools
from collections import Counter
import sys
import math
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.stats import chi2
from numba import njit

@njit
def count_block_codes(codes, n_unique):
    counts = np.zeros(n_unique, dtype=np.int64)
    for i in range(codes.shape[0]):
        counts[codes[i]] += 1
    return counts

class RandomnessAnalysis:
    def __init__(self, blocks_df, s):
        self.blocks_df = blocks_df
        self.s = s
        self.n_blocks = len(blocks_df)
        self.k = self.blocks_df.shape[1] 


    def compute_blocks_frequencies(self):
        """
        Version optimisée avec Numba.
        """
        # Conversion en numpy array
        blocks = self.blocks_df.values.astype(np.int32)

        # Encodage des blocs en nombres uniques
        powers = (self.s ** np.arange(self.k - 1, -1, -1)).astype(np.int32)
        codes = np.dot(blocks, powers)

        # Comptage avec Numba
        max_code = self.s ** self.k
        counts = count_block_codes(codes, max_code)

        # Préparation du DataFrame (comme avant)
        all_combinations = np.arange(max_code)
        relative_freq = counts / self.n_blocks if self.n_blocks > 0 else np.zeros_like(counts)
        absolute_freq = counts

        self.frequencies = pd.DataFrame(
            {
                "block": all_combinations,
                "absolute frequency": absolute_freq,
                "relative frequency": relative_freq
            }
        )

        return self.frequencies

    def shannon_entropy(self):
        if not hasattr(self, 'frequencies'):
            self.compute_blocks_frequencies()
        H = 0.0
        for i in range(self.s**self.k):
            p = self.frequencies.iloc[i, 2]
            if p > 0:
                H += -p * math.log(p)
        return H
    def entropy_bias_test(self, m=1):
        B = 2 * self.n_blocks * (self.k * math.log(self.s) - self.shannon_entropy())
        df = self.s**self.k - 1
        alpha_90 = 1 - (1 - 0.1) ** (1 / m)
        alpha_95 = 1 - (1 - 0.05) ** (1 / m)
        alpha_99 = 1 - (1 - 0.01) ** (1 / m)
        quantile_90 = chi2.ppf(1-alpha_90, df)
        quantile_95 = chi2.ppf(1-alpha_95, df)
        quantile_99 = chi2.ppf(1-alpha_99, df)
        hypothesis = B > quantile_99
        p_value = chi2.sf(B, df)

        result = pd.DataFrame(
            {
                "Bias": [B],
                "Quantile 90%": [quantile_90],
                "Quantile 95%": [quantile_95],
                "Quantile 99%": [quantile_99],
                "P-value": [p_value],
                "Mean": [df],
                "Hypothesis 1": [hypothesis]
            }
        )
        result = result.T
        result.columns = ["Entropy Bias test"]
        return result
        
    def KL_divergence(self):
        df = self.blocks_df

        block_cols = df.columns[:-1]
        target_col = df.columns[-1]

        n_blocks = len(df)
        
        joint_counts = df.groupby(list(block_cols) + [target_col]).size().reset_index(name='count_ij')
        f_i_dot = df.groupby(list(block_cols)).size().reset_index(name='count_i')
        f_dot_j = df.groupby(target_col).size().reset_index(name='count_j')

        merged = joint_counts.merge(f_i_dot, on=list(block_cols))
        merged = merged.merge(f_dot_j, on=target_col)

        # D += 2 * count_ij * log((n_blocks * count_ij) / (count_i * count_j))
        merged['term'] = 2 * merged['count_ij'] * np.log(
            (n_blocks * merged['count_ij']) / (merged['count_i'] * merged['count_j'])
        )

        D = merged['term'].sum()
        return D

    
    def KL_divergence_test(self,m=1):

        D = self.KL_divergence()
        df = (self.s**(self.k-1) - 1) * (self.s - 1)

        alpha_90 = 1 - (1 - 0.1) ** (1 / m)
        alpha_95 = 1 - (1 - 0.05) ** (1 / m)
        alpha_99 = 1 - (1 - 0.01) ** (1 / m)
        
        quantile_90 = chi2.ppf(1-alpha_90, df)
        quantile_95 = chi2.ppf(1-alpha_95, df)
        quantile_99 = chi2.ppf(1-alpha_99, df)
        hypothesis = D > quantile_99
        p_value = chi2.sf(D, df)

        result = pd.DataFrame(
            {
                "NP Statistic": [D],
                "Quantile 90%": [quantile_90],
                "Quantile 95%": [quantile_95],
                "Quantile 99%": [quantile_99],
                "P-value": [p_value],
                "Mean": [df],
                "Hypothesis 1": [hypothesis]
            }
        )

        result = result.T
        result.columns = ["NP Statistic test"]
        return result



if __name__ == "__main__":
    df = pd.read_csv("data/blocks/BTCUSDT_blocks_3.csv", header=None)
    nb_symbols = 2
    test = RandomnessAnalysis(df, s=nb_symbols)

    test_divergence = test.KL_divergence_test()
    print(test_divergence)

    """
    frequencies_df = test.compute_blocks_frequencies()
    test_entropy = test.entropy_bias_test()
    print(test_entropy)
    plot_block_frequencies(frequencies_df)
    """