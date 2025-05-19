import numpy as np
import pandas as pd
import gc
from data.DataManager import DataManager
from main.RandomnessAnalysis import RandomnessAnalysis
from main.CrossRandomnessAnalysis import CrossRandomnessAnalysis
from utils.VisualizationTools import plot_3D
from tqdm import tqdm


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
                           test='Entropy Bias',
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

        for i in range(1, max_block_size + 1, step):
            data_manager = DataManager([self.asset], self.symbols, year, month, day, aggregation_level)
            blocks_df = data_manager.block_constructor(block_size=i, overlapping=self.overlapping)[self.asset]

            analysis = RandomnessAnalysis(blocks_df=blocks_df, s=self.s)
            _ = analysis.compute_blocks_frequencies()

            if test == 'Entropy Bias':
                test_result = analysis.entropy_bias_test()
            elif test == 'NP Statistic' and i < 2:
                print(f"[WARNING] KL Divergence test is not defined for block size {i}. Skipping.")
                continue
            else:
                raise ValueError("Invalid test type.")

            self.results_by_block_size['Block size'].append(i)
            self.results_by_block_size['Test statistic'].append(test_result.iloc[0, 0])
            self.results_by_block_size['Quantile 90'].append(test_result.iloc[1, 0])
            self.results_by_block_size['Quantile 95'].append(test_result.iloc[2, 0])
            self.results_by_block_size['Quantile 99'].append(test_result.iloc[3, 0])
            self.results_by_block_size['Mean'].append(test_result.iloc[5, 0])

            # Nettoyage mémoire
            del data_manager, blocks_df, analysis, test_result
            gc.collect()

        df = pd.DataFrame(self.results_by_block_size)
        df.set_index('Block size', inplace=True)
        return df

    def test_by_aggregation_level(self,
                                   test='Entropy Bias',
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
            blocks_df = data_manager.block_constructor(block_size=block_size, overlapping=self.overlapping)[self.asset]

            analysis = RandomnessAnalysis(blocks_df=blocks_df, s=self.s)
            _ = analysis.compute_blocks_frequencies()

            if test == 'Entropy Bias':
                test_result = analysis.entropy_bias_test()
            elif test == 'NP Statistic':
                test_result = analysis.KL_divergence_test()
            else:
                raise ValueError("Invalid test type.")

            self.results_by_aggregation_level['Aggregation level'].append(i)
            self.results_by_aggregation_level['Test statistic'].append(test_result.iloc[0, 0])
            self.results_by_aggregation_level['Quantile 90'].append(test_result.iloc[1, 0])
            self.results_by_aggregation_level['Quantile 95'].append(test_result.iloc[2, 0])
            self.results_by_aggregation_level['Quantile 99'].append(test_result.iloc[3, 0])
            self.results_by_aggregation_level['Mean'].append(test_result.iloc[5, 0])

            # Nettoyage mémoire
            del data_manager, blocks_df, analysis, test_result
            gc.collect()

        df = pd.DataFrame(self.results_by_aggregation_level)
        df.set_index('Aggregation level', inplace=True)
        return df

    def plot_3D_test_result(self,
                            asset=None,
                            test='Entropy Bias',
                            max_block_size=10,
                            max_aggregation_level=50,
                            step_block=1,
                            step_aggregation=1,
                            year=2024,
                            month=11,
                            day=None):

        if asset is None:
            asset = self.asset

        result_3D = np.zeros((max_aggregation_level,
                            max_block_size,
                            2))

        total = ((max_aggregation_level - 1) // step_aggregation + 1) * \
                ((max_block_size - 1) // step_block + 1)

        pbar = tqdm(total=total, desc=f"[{asset}] Grid test 3D")

        for i in range(1, max_aggregation_level + 1, step_aggregation):
            for j in range(1, max_block_size + 1, step_block):
                data_manager = DataManager([asset], self.symbols, aggregation_level=i,
                                        year=year, month=month, day=day)
                blocks_df = data_manager.block_constructor(block_size=j, overlapping=self.overlapping)[asset]

                analysis = RandomnessAnalysis(blocks_df=blocks_df, s=self.s)
                _ = analysis.compute_blocks_frequencies()

                if test == 'Entropy Bias':
                    test_result = analysis.entropy_bias_test()
                elif test == 'NP Statistic':
                    test_result = analysis.KL_divergence_test()
                else:
                    raise ValueError("Invalid test type.")

                stat = test_result.iloc[0, 0]
                quantile_99 = test_result.iloc[3, 0]

                result_3D[i - 1, j - 1, 0] = stat
                result_3D[i - 1, j - 1, 1] = quantile_99

                del data_manager, blocks_df, analysis, test_result
                gc.collect()
                pbar.update(1)

        pbar.close()
        return plot_3D(result_3D, test)


class CrossMultiTester:
    def __init__(self,
                 asset_context: str,
                 asset_target: str,
                 symbols: dict = DEFAULT_SYMBOLS):
        self.asset_context = asset_context
        self.asset_target = asset_target
        self.symbols = symbols
        self.s = len(symbols)

    def test_by_block_size(self,
                           test='Entropy Bias',
                           max_block_size=10,
                           step=1,
                           alpha_context=1,
                           year=2024,
                           month=11,
                           day=None) -> pd.DataFrame:

        results = {
            'Block size': [],
            'Alpha target': [],
            'Test statistic': [],
            'Quantile 90': [],
            'Quantile 95': [],
            'Quantile 99': [],
            'Mean': []
        }

        for k in range(1, max_block_size + 1, step):
            dm_ctx = DataManager([self.asset_context], self.symbols,
                                 year, month, day,
                                 aggregation_level=alpha_context)

            n_blocks_ctx = len(dm_ctx.datasets[self.asset_context]['symbol'])
            alpha_tgt = dm_ctx.matching_aggregation_for(
                self.asset_target, blocks=n_blocks_ctx,
                year=year, month=month, day=day
            )

            dm_tgt = DataManager([self.asset_target], self.symbols,
                                 year, month, day,
                                 aggregation_level=alpha_tgt)

            cra = CrossRandomnessAnalysis(
                symbols_context=dm_ctx.datasets[self.asset_context]['symbol'],
                symbols_target=dm_tgt.datasets[self.asset_target]['symbol'],
                k=k, s=self.s,
                asset_context=self.asset_context,
                alpha_context=alpha_context,
                asset_target=self.asset_target,
                alpha_target=alpha_tgt
            )

            res = cra.entropy_bias_test() if test == 'Entropy Bias' else cra.KL_divergence_test()

            results['Block size'].append(k)
            results['Alpha target'].append(alpha_tgt)
            results['Test statistic'].append(res.iloc[0, 0])
            results['Quantile 90'].append(res.iloc[1, 0])
            results['Quantile 95'].append(res.iloc[2, 0])
            results['Quantile 99'].append(res.iloc[3, 0])
            results['Mean'].append(res.iloc[5, 0])

            del dm_ctx, dm_tgt, cra, res
            gc.collect()

        return pd.DataFrame(results).set_index('Block size')

    def test_by_aggregation_level(self,
                                  test='Entropy Bias',
                                  aggregation_levels=[1, 2, 5, 10],
                                  k=2,
                                  year=2024,
                                  month=11,
                                  day=None) -> pd.DataFrame:

        results = {
            'Aggregation level': [],
            'Alpha target': [],
            'Test statistic': [],
            'Quantile 90': [],
            'Quantile 95': [],
            'Quantile 99': [],
            'Mean': []
        }

        for alpha_ctx in aggregation_levels:
            dm_ctx = DataManager([self.asset_context], self.symbols,
                                 year, month, day,
                                 aggregation_level=alpha_ctx)

            n_blocks_ctx = len(dm_ctx.datasets[self.asset_context]['symbol'])
            alpha_tgt = dm_ctx.matching_aggregation_for(
                self.asset_target, blocks=n_blocks_ctx,
                year=year, month=month, day=day
            )

            dm_tgt = DataManager([self.asset_target], self.symbols,
                                 year, month, day,
                                 aggregation_level=alpha_tgt)

            cra = CrossRandomnessAnalysis(
                symbols_context=dm_ctx.datasets[self.asset_context]['symbol'],
                symbols_target=dm_tgt.datasets[self.asset_target]['symbol'],
                k=k, s=self.s,
                asset_context=self.asset_context,
                alpha_context=alpha_ctx,
                asset_target=self.asset_target,
                alpha_target=alpha_tgt
            )

            res = cra.entropy_bias_test() if test == 'Entropy Bias' else cra.KL_divergence_test()

            results['Aggregation level'].append(alpha_ctx)
            results['Alpha target'].append(alpha_tgt)
            results['Test statistic'].append(res.iloc[0, 0])
            results['Quantile 90'].append(res.iloc[1, 0])
            results['Quantile 95'].append(res.iloc[2, 0])
            results['Quantile 99'].append(res.iloc[3, 0])
            results['Mean'].append(res.iloc[5, 0])

            del dm_ctx, dm_tgt, cra, res
            gc.collect()

        return pd.DataFrame(results).set_index('Aggregation level')

    def test_grid(self,
                  test='Entropy Bias',
                  list_aggregations=[1, 2, 5, 10, 20],
                  list_block_sizes=[1, 2, 3, 4, 5],
                  year=2024,
                  month=11,
                  day=None):

        result_3D = np.zeros((max(list_aggregations),
                              max(list_block_sizes),
                              2))

        total = len(list_aggregations) * len(list_block_sizes)
        pbar = tqdm(total=total, desc=f"[{self.asset_context}→{self.asset_target}] Cross test grid")

        for alpha_ctx in list_aggregations:
            dm_ctx = DataManager([self.asset_context], self.symbols,
                                 year, month, day,
                                 aggregation_level=alpha_ctx)

            n_blocks_ctx = len(dm_ctx.datasets[self.asset_context]['symbol'])
            alpha_tgt = dm_ctx.matching_aggregation_for(
                self.asset_target, blocks=n_blocks_ctx,
                year=year, month=month, day=day
            )

            dm_tgt = DataManager([self.asset_target], self.symbols,
                                 year, month, day,
                                 aggregation_level=alpha_tgt)

            for k in list_block_sizes:
                cra = CrossRandomnessAnalysis(
                    symbols_context=dm_ctx.datasets[self.asset_context]['symbol'],
                    symbols_target=dm_tgt.datasets[self.asset_target]['symbol'],
                    k=k, s=self.s,
                    asset_context=self.asset_context,
                    alpha_context=alpha_ctx,
                    asset_target=self.asset_target,
                    alpha_target=alpha_tgt
                )

                res = cra.entropy_bias_test() if test == 'Entropy Bias' else cra.KL_divergence_test()

                stat = res.iloc[0, 0]
                quantile_99 = res.iloc[3, 0]

                result_3D[alpha_ctx - 1, k - 1, 0] = stat
                result_3D[alpha_ctx - 1, k - 1, 1] = quantile_99

                del cra, res
                gc.collect()
                pbar.update(1)

            del dm_ctx, dm_tgt
            gc.collect()

        pbar.close()
        return result_3D

    @staticmethod
    def to_dataframe_from_grid(result_3D, aggregation_levels, block_sizes, test_name="Test"):
        records = []
        for i, alpha in enumerate(aggregation_levels):
            for j, k in enumerate(block_sizes):
                stat = result_3D[i, j, 0]
                q99 = result_3D[i, j, 1]
                records.append({
                    'Aggregation level': alpha,
                    'Block size': k,
                    f'{test_name} statistic': stat,
                    'Quantile 99': q99
                })
        return pd.DataFrame(records)
