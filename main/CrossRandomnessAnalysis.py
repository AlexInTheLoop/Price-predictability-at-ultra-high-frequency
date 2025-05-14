import pandas as pd
from main.RandomnessAnalysis import RandomnessAnalysis
from utils.BlockConstructors import cross_overlapping_blocks

class CrossRandomnessAnalysis:
    def __init__(self, 
                 blocks_A,
                 blocks_B, 
                 k,    
                 s):
        
        self.blocks_A = blocks_A
        self.blocks_B = blocks_B
        self.k        = k
        self.s        = s

        self.cross_df = pd.DataFrame(
            cross_overlapping_blocks(
                blocks_B['symbol'].values,
                blocks_A['symbol'].values,
                k
            )
        )

    def entropy_bias_test(self):
        analyser = RandomnessAnalysis(self.cross_df, s=self.s)
        return analyser.entropy_bias_test()

    def KL_divergence_test(self):
        analyser = RandomnessAnalysis(self.cross_df, s=self.s)
        return analyser.KL_divergence_test()