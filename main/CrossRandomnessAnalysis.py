import pandas as pd
from main.RandomnessAnalysis import RandomnessAnalysis
from utils.BlockConstructors import (
    cross_overlapping_blocks,
    cross_non_overlapping_blocks
)

class CrossRandomnessAnalysis:
    def __init__(self,
                 blocks_context,
                 blocks_target,
                 k,
                 s,
                 asset_context,
                 alpha_context,
                 asset_target,
                 alpha_target):
        
        self.blocks_ctx    = blocks_context.copy()
        self.blocks_tgt    = blocks_target.copy()
        self.k             = k
        self.s             = s
        self.asset_context = asset_context
        self.alpha_context = alpha_context
        self.asset_target  = asset_target
        self.alpha_target  = alpha_target

        # prepare overlapping cross-blocks (sliding windows)
        arr_ovlp = cross_overlapping_blocks(
            self.blocks_ctx['symbol'].values,
            self.blocks_tgt['symbol'].values,
            k
        )
        self.cross_df_ovlp = pd.DataFrame(arr_ovlp)

        # prepare non-overlapping cross-blocks (disjoint windows)
        arr_non = cross_non_overlapping_blocks(
            self.blocks_ctx['symbol'].values,
            self.blocks_tgt['symbol'].values,
            k
        )
        self.cross_df_non  = pd.DataFrame(arr_non)

    def compute_cross_frequencies(self) -> pd.DataFrame:
        # Context frequencies (blocks of size k)
        freqs_ctx = RandomnessAnalysis(self.blocks_ctx, s=self.s).compute_blocks_frequencies()
        freqs_ctx['asset']      = self.asset_context
        freqs_ctx['alpha']      = self.alpha_context
        freqs_ctx['block_size'] = self.k

        # Target frequencies (blocks of size 1)
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
