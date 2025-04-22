import pandas as pd
import itertools
from collections import Counter
import sys
import math
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.stats import chi2
from utils.VisualizationTools import plot_block_frequencies

class RandomnessAnalysis:
    def __init__(self, blocks_df, s, k):
        self.blocks_df = blocks_df
        self.s = s
        self.k = k
        self.n_blocks = len(blocks_df)

    def compute_blocks_frequencies(self):
        symbols = list(range(self.s))
        k = self.blocks_df.shape[1] 
        all_combinations = list(itertools.product(symbols, repeat=k)) 

        observed_blocks = [tuple(row) for row in self.blocks_df.values]

        block_counts = Counter(observed_blocks)

        relative_frequencies = {
            combination: block_counts[combination] / self.n_blocks if self.n_blocks > 0 else 0
            for combination in all_combinations
        }

        absolute_frequencies = {
            combination: block_counts[combination] if combination in block_counts else 0
            for combination in all_combinations
        }

        self.frequencies = pd.DataFrame(
            {
                "block": list(relative_frequencies.keys()),
                "absolute frequency": list(absolute_frequencies.values()),
                "relative frequency": list(relative_frequencies.values())
            }
        )

        return self.frequencies

    def shannon_entropy(self):
        H = 0.0
        for i in range(self.s**self.k):
            p = self.frequencies.iloc[i, 2]
            if p > 0:
                H += -p * math.log(p)
        return H

    def entropy_bias_test(self):
        B = 2 * self.n_blocks * (self.k * math.log(self.s) - self.shannon_entropy())
        df = self.s**self.k - 1
        quantile_90 = chi2.ppf(0.90, df)
        quantile_95 = chi2.ppf(0.95, df)
        quantile_99 = chi2.ppf(0.99, df)
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
        f_ij = Counter()
        f_i_dot = Counter()
        f_dot_j = Counter()

        for i in range(self.n_blocks):
            block_i = tuple(self.blocks_df.iloc[i,:-1].values.flatten())
            symbol_j = self.blocks_df.iloc[i, -1]
            f_ij[(block_i, symbol_j)] += 1
            f_i_dot[block_i] += 1
            f_dot_j[symbol_j] += 1
        
        D = 0.0
        for (block_i, symbol_j), count_ij in f_ij.items():
            count_i = f_i_dot[block_i]
            count_j = f_dot_j[symbol_j]
            if count_ij > 0 and count_i > 0 and count_j > 0:
                D += 2 * count_ij * math.log( (self.n_blocks * count_ij) / (count_i * count_j) )

        return D
    
    def KL_divergence_test(self):

        D = self.KL_divergence()
        df = (self.s**(self.k-1) - 1) * (self.s - 1)
        
        quantile_90 = chi2.ppf(0.90, df)
        quantile_95 = chi2.ppf(0.95, df)
        quantile_99 = chi2.ppf(0.99, df)
        hypothesis = D > quantile_99
        p_value = chi2.sf(D, df)

        result = pd.DataFrame(
            {
                "KL Divergence": [D],
                "Quantile 90%": [quantile_90],
                "Quantile 95%": [quantile_95],
                "Quantile 99%": [quantile_99],
                "P-value": [p_value],
                "Mean": [df],
                "Hypothesis 1": [hypothesis]
            }
        )

        result = result.T
        result.columns = ["KL Divergence test"]
        return result



if __name__ == "__main__":
    df = pd.read_csv("data/blocks/BTCUSDT_blocks_3.csv", header=None)
    nb_symbols = 2
    block_size = 3
    test = RandomnessAnalysis(df, s=nb_symbols, k=block_size)

    test_divergence = test.KL_divergence_test()
    print(test_divergence)

    """
    frequencies_df = test.compute_blocks_frequencies()
    test_entropy = test.entropy_bias_test()
    print(test_entropy)
    plot_block_frequencies(frequencies_df)
    """