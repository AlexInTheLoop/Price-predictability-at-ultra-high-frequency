import numpy as np
import pandas as pd
from main.RandomnessAnalysis import RandomnessAnalysis
import scipy.stats as stats

DEFAULT_SYMBOLS = {
                    0: [(-np.inf, 0), (False, False)],
                    1: [(0, np.inf), (False, False)]
                }

class PredictableDayAnalysis:
    def __init__(self,
                 pair,
                 data_manager):
        self.pair = pair
        self.df = data_manager.datasets[pair]
        self.s = len(data_manager.symbols.keys())
        self.df.index = pd.to_datetime(self.df.index,unit='ms')
        self.df.reset_index(drop=False, inplace=True)
        self.daily_df = self._split__by_day(self.df)
    
    @staticmethod
    def _split__by_day(df):
        daily_df = {}
        grouped = df.groupby(df['timestamp'].dt.date)
        for i, (date, group) in enumerate(grouped,start=1):
            daily_df[f"day {i}"] = group.reset_index(drop=True)
        return daily_df
    
    @staticmethod
    def _detect_jump(logprices):
        returns = logprices[1:] - logprices[:-1]
        n = len(returns)
        KJ = int(np.ceil(np.sqrt(n)))
        BV = np.array([
            np.dot(np.abs(returns[i-KJ+2:i]), np.abs(returns[i-KJ+1:i-1]))
            for i in range(KJ-1, n)
        ]) / (KJ-2)
        returns = returns[KJ-1:]
        BV = BV[BV > 0]
        returns = returns[:len(BV)]
        LJ = returns / np.sqrt(BV)
        c = np.sqrt(2 / np.pi)
        Cn = np.sqrt(2 * np.log(n)) / c - (np.log(np.pi) + np.log(np.log(n))) / (2 * c * np.sqrt(2 * np.log(n)))
        Sn = 1 / (c * np.sqrt(2 * np.log(n)))
        beta = -np.log(-np.log(0.99))
        test = (np.abs(LJ) - Cn) / Sn
        return np.sum(test > beta) / len(test)
    
    def analyze_days(self,blocks):
        efficient_results = []
        inefficient_results = []
        n = len(self.daily_df)

        for i, (day, df) in enumerate(self.daily_df.items()):
            print(f"{i+1}/{n}")

            jumps_frac = self._detect_jump(df['price'].values)
            non_zero_returns = df['returns'][df['returns'] != 0]
            squared_returns = non_zero_returns ** 2
            autocorr = non_zero_returns.autocorr(lag=1)
            vol_autocorr = squared_returns.autocorr(lag=1)
            nu, loc, scale = stats.t.fit(df['returns'])
            mean_return = df['returns'].mean()
            zero_fraction = (df['returns'] == 0).sum() / len(df['returns'])

            analysis = RandomnessAnalysis(blocks_df=blocks, s=self.s)
            result = analysis.KL_divergence_test()
            stat = result.iloc[0, 0]
            quantile_99 = result.iloc[3, 0]
            hypothesis = result.iloc[6, 0]

            result = {
                'Jump fraction': jumps_frac,
                'Autocorrelation': autocorr,
                'Autocorrelation of squared returns': vol_autocorr,
                'Student distribution degree of freedom': nu,
                'Student distribution mean': loc,
                'Student distribution standard deviation': scale,
                'Returns mean': mean_return,
                'Fractions of zero-returns': zero_fraction,
                'NP Statistic statistics': stat,
                'Empirical quantile': quantile_99
            }

            if hypothesis:
                inefficient_results.append(result)
            else:
                efficient_results.append(result)

        self.efficient_df = pd.DataFrame(efficient_results)
        self.inefficient_df = pd.DataFrame(inefficient_results)