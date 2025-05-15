import numpy as np
import pandas as pd
from main.RandomnessAnalysis import RandomnessAnalysis
import scipy.stats as stats
import multiprocessing as mp
from tqdm import tqdm


DEFAULT_SYMBOLS = {
    0: [(-np.inf, 0), (False, False)],
    1: [(0, np.inf), (False, False)]
}
class PredictableDayAnalysis:
    def __init__(self, pair, data_manager):
        self.pair = pair
        self.df = data_manager.datasets[pair]
        self.s = len(data_manager.symbols.keys())
        self.df.index = pd.to_datetime(self.df.index, unit='ms')
        self.df.reset_index(drop=False, inplace=True)
        self.daily_df = self._split_by_day(self.df)

    @staticmethod
    def _split_by_day(df):
        daily_df = {}
        grouped = df.groupby(df['timestamp'].dt.date)
        for i, (date, group) in enumerate(grouped, start=1):
            daily_df[f"day {i}"] = group.reset_index(drop=True)
        return daily_df

    @staticmethod
    def _detect_jump(logprices):
        returns = logprices[1:] - logprices[:-1]
        n = len(returns)
        if n < 2:
            return 0
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

    @staticmethod
    def _process_day(args):
        day, df_day, s, block_size = args
        results_day = []

        symbols = df_day['symbol'].values
        if len(symbols) < block_size:
            return results_day

        n = symbols.shape[0]
        shape = (n - block_size + 1, block_size)
        strides = (symbols.strides[0], symbols.strides[0])
        blocks = np.lib.stride_tricks.as_strided(symbols, shape=shape, strides=strides)

        analysis = RandomnessAnalysis(pd.DataFrame(blocks), s)
        result = analysis.KL_divergence_test()

        stat = result.iloc[0, 0]
        quantile_99 = result.iloc[3, 0]
        hypothesis = result.iloc[6, 0]

        jumps_frac = PredictableDayAnalysis._detect_jump(df_day['price'].values)
        non_zero_returns = df_day['returns'][df_day['returns'] != 0]
        squared_returns = non_zero_returns ** 2
        autocorr = non_zero_returns.autocorr(lag=1) if len(non_zero_returns) > 1 else np.nan
        vol_autocorr = squared_returns.autocorr(lag=1) if len(squared_returns) > 1 else np.nan
        try:
            nu, loc, scale = stats.t.fit(df_day['returns'])
        except Exception:
            nu, loc, scale = np.nan, np.nan, np.nan
        mean_return = df_day['returns'].mean()
        zero_fraction = (df_day['returns'] == 0).sum() / len(df_day)

        metrics = {
            'Jump fraction': jumps_frac,
            'Autocorrelation': autocorr,
            'Autocorrelation of squared returns': vol_autocorr,
            'Student distribution degree of freedom': nu,
            'Student distribution mean': loc,
            'Student distribution standard deviation': scale,
            'Returns mean': mean_return,
            'Fractions of zero-returns': zero_fraction,
            'KL Divergence statistics': stat,
            'Empirical quantile': quantile_99,
            'Hypothesis': hypothesis
        }

        return [metrics]

    def analyze_days(self, block_size=2, n_jobs=None):
        print(f"[SYSTEM] Starting analysis → {len(self.daily_df)} days to process...")
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        args_list = [(day, df, self.s, block_size) for day, df in self.daily_df.items()]

        results = []
        with mp.Pool(processes=n_jobs) as pool:
            for output in tqdm(pool.imap_unordered(PredictableDayAnalysis._process_day, args_list),
                               total=len(args_list),
                               desc=f"[{self.pair}] Analyzing days"):
                results.extend(output)

        efficient_results = []
        inefficient_results = []
        for res in results:
            if res['Hypothesis']:
                inefficient_results.append(res)
            else:
                efficient_results.append(res)

        self.efficient_df = pd.DataFrame(efficient_results)
        self.inefficient_df = pd.DataFrame(inefficient_results)

        print(f"[SYSTEM] Analysis completed : {len(results)} days analyzed → "
              f"{len(self.efficient_df)} efficient, {len(self.inefficient_df)} inefficient.")
