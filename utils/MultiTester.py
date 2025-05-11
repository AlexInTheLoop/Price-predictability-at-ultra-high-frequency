from data.DataManager import DataManager
from main.RandomnessAnalysis import RandomnessAnalysis
import pandas as pd
import numpy as np
from utils.VisualizationTools import plot_3D_entropy_bias

DEFAULT_SYMBOLS = {
                    0: [(-np.inf, 0), (False, False)],
                    1: [(0, np.inf), (False, False)]
                }

class MultiTester:
    def __init__(self, asset, symbols=DEFAULT_SYMBOLS, overlapping=False):
        self.asset = asset
        self.symbols = symbols
        self.overlapping = overlapping
        self.s = len(symbols.keys())
    
    def test_by_block_size(self,
                           test ='Entropy Bias', 
                           max_block_size=10, 
                           step=1, 
                           aggregation_level=1,
                           year=2024,
                           month=11,
                           day=None):
        
        self.results_by_block_size = {
            'Block size': [],
            'Test statistic': [],
            'Quantile 99': [],
            'Quantile 95': [],
            'Quantile 90': [],
            'Mean': []
        }
        data_manager = DataManager([self.asset], self.symbols, year, month, day, aggregation_level)

        for i in range(1, max_block_size + 1, step):
            blocks = data_manager.block_constructor(block_size=i, overlapping=self.overlapping)
            blocks = blocks[self.asset]

            analysis = RandomnessAnalysis(blocks_df=blocks, s=self.s)
            _ = analysis.compute_blocks_frequencies()
            if test == 'Entropy Bias':
                test_result = analysis.entropy_bias_test()
            elif test == 'KL Divergence':
                test_result = analysis.KL_divergence_test()
            else:
                raise ValueError("Invalid test type. Use 'Entropy Bias' or 'KL Divergence'.")

            self.results_by_block_size['Block size'].append(i)
            self.results_by_block_size['Test statistic'].append(test_result.iloc[0, 0])
            self.results_by_block_size['Quantile 90'].append(test_result.iloc[1, 0])
            self.results_by_block_size['Quantile 95'].append(test_result.iloc[2, 0])
            self.results_by_block_size['Quantile 99'].append(test_result.iloc[3, 0])
            self.results_by_block_size['Mean'].append(test_result.iloc[5, 0])

        df = pd.DataFrame(self.results_by_block_size)
        df.set_index('Block size', inplace=True)
        return df
    
    def test_by_aggregation_level(self,test='Entropy Bias',
                                  max_aggregation_level=50,
                                  step=1,
                                  block_size=2,
                                  year=2024,
                                  month=11,
                                  day=None):
        self.results_by_aggregation_level = {
            'Aggregation level': [],
            'Test statistic': [],
            'Quantile 90': [],
            'Quantile 95': [],
            'Quantile 99': [],
            'Mean': []
        }

        for i in range(1, max_aggregation_level + 1, step):
            data_manager = DataManager([self.asset], self.symbols, aggregation_level=i, year=year, month=month, day=day)
            blocks = data_manager.block_constructor(block_size=block_size, overlapping=self.overlapping)
            blocks = blocks[self.asset]

            analysis = RandomnessAnalysis(blocks_df=blocks, s=self.s)
            _ = analysis.compute_blocks_frequencies()
            if test == 'Entropy Bias':
                test_result = analysis.entropy_bias_test()
            elif test == 'KL Divergence':
                test_result = analysis.KL_divergence_test()
            else:
                raise ValueError("Invalid test type. Use 'Entropy Bias' or 'KL Divergence'.")

            self.results_by_aggregation_level['Aggregation level'].append(i)
            self.results_by_aggregation_level['Test statistic'].append(test_result.iloc[0, 0])
            self.results_by_aggregation_level['Quantile 90'].append(test_result.iloc[1, 0])
            self.results_by_aggregation_level['Quantile 95'].append(test_result.iloc[2, 0])
            self.results_by_aggregation_level['Quantile 99'].append(test_result.iloc[3, 0])
            self.results_by_aggregation_level['Mean'].append(test_result.iloc[5, 0])

        df = pd.DataFrame(self.results_by_aggregation_level)
        df.set_index('Aggregation level', inplace=True)
        return df
    
    def plot_3D_test_result(self,asset='BTCUSDT',
                            test = 'Entropy Bias', 
                            max_block_size=10,
                            max_aggregation_level=50,
                            step_block=1,
                            step_aggregation=1,
                            year=2024,
                            month=11,
                            day=None):
        result_3D = np.zeros((max_aggregation_level,
                              max_block_size, 
                              2))

        for i in range(1, max_aggregation_level + 1, step_aggregation):
            for j in range(1, max_block_size + 1, step_block):
                data_manager = DataManager([asset], self.symbols, aggregation_level=i, year=year, month=month, day=day)
                blocks = data_manager.block_constructor(block_size=j, overlapping=False)
                blocks_btc = blocks[asset]
                analysis = RandomnessAnalysis(blocks_df=blocks_btc, s=2)
                _ = analysis.compute_blocks_frequencies()

                if test == 'Entropy Bias':
                    test_result = analysis.entropy_bias_test()
                elif test == 'KL Divergence':
                    test_result = analysis.KL_divergence_test()
                else:
                    raise ValueError("Invalid test type. Use 'Entropy Bias' or 'KL Divergence'.")

                stat = test_result.iloc[0, 0]
                quantile_99 = test_result.iloc[3, 0]

                result_3D[i - 1, j - 1, 0] = stat
                result_3D[i - 1, j - 1, 1] = quantile_99
        plot_3D_entropy_bias(result_3D,test)
        
