import pandas as pd
import numpy as np
from utils.BlockConstructors import overlapping_blocks
from main.RandomnessAnalysis import RandomnessAnalysis
from utils.BlockConstructors import (
    cross_overlapping_blocks,
    cross_non_overlapping_blocks
)

class CrossRandomnessAnalysis:
    def __init__(self,
                 symbols_context,
                 symbols_target,
                 k,
                 s,
                 asset_context,
                 alpha_context,
                 asset_target,
                 alpha_target):
        
        self.symbols_ctx = np.asarray(symbols_context, dtype=np.int8)
        self.symbols_tgt = np.asarray(symbols_target, dtype=np.int8)
        self.k             = k
        self.s             = s
        self.asset_context = asset_context
        self.alpha_context = alpha_context
        self.asset_target  = asset_target
        self.alpha_target  = alpha_target
        

        # prepare overlapping cross-blocks (sliding windows)
        arr_ovlp = cross_overlapping_blocks(self.symbols_ctx, self.symbols_tgt, self.k)
        self.cross_df_ovlp = pd.DataFrame(arr_ovlp)

        # prepare non-overlapping cross-blocks (disjoint windows)
        arr_non = cross_non_overlapping_blocks(self.symbols_ctx, self.symbols_tgt, self.k)
        self.cross_df_non  = pd.DataFrame(arr_non)        
        self.blocks_ctx = pd.DataFrame(overlapping_blocks(self.symbols_ctx, self.k))
        self.blocks_tgt = pd.DataFrame(overlapping_blocks(self.symbols_tgt, 1))
        
    def compute_cross_frequencies(self) -> pd.DataFrame:
        freqs_ctx = RandomnessAnalysis(self.blocks_ctx, s=self.s).compute_blocks_frequencies()
        freqs_ctx['asset']      = self.asset_context
        freqs_ctx['alpha']      = self.alpha_context
        freqs_ctx['block_size'] = self.k

        freqs_tgt = RandomnessAnalysis(self.blocks_tgt, s=self.s).compute_blocks_frequencies()
        freqs_tgt['asset']      = self.asset_target
        freqs_tgt['alpha']      = self.alpha_target
        freqs_tgt['block_size'] = 1

        return pd.concat([freqs_ctx, freqs_tgt], ignore_index=True)

    def entropy_bias_test(self) -> pd.DataFrame:
        analyser = RandomnessAnalysis(self.cross_df_non, s=self.s)
        return analyser.entropy_bias_test()

    def KL_divergence_test(self) -> pd.DataFrame:
        analyser = RandomnessAnalysis(self.cross_df_ovlp, s=self.s)
        return analyser.KL_divergence_test()
