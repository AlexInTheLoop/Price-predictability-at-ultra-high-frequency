import math
import numpy as np
import pandas as pd
from data.DataCollectors import HistoricalDataCollectorParquet, HistoricalDataCollector
from data.DataManager import DataManager
from utils.BlockConstructors import non_overlapping_blocks, overlapping_blocks
from main.RandomnessAnalysis import RandomnessAnalysis
from utils.VisualizationTools import sub_plots_comp
import os
from data.DataManager import PREPROCCESSED_DATA_FOLDER


def get_assets_properties(pairs, s, year, month, day=None, aggregation_level=1):
        if year is None or month is None:
            when = None
        elif day is None:
            when = f"{year}-" + f"{int(month):02d}"
        else:
            when = f"{year}-" + f"{int(month):02d}-" + f"{int(day):02d}"

        result = {}
        
        for pair in pairs:          
            filename = os.path.join(PREPROCCESSED_DATA_FOLDER,
                f"{pair.upper()}_processed"
                + (f"_{when}" if when is not None else "_REAL_TIME")
                + f"_S={s}"
                + f"_A={aggregation_level}.parquet"  
            )
            if not os.path.exists(filename):
              raise FileNotFoundError(f"File {filename} does not exist. Launch the data collector/manager first.")
            df = pd.read_parquet(filename)

            mean_price = np.exp(df["price"]).mean()
            std_price = np.exp(df["price"]).std()
            mean_return = df["returns"].mean()
            std_return = df["returns"].std()
            volume = df['volume'].sum()
            mean_volume = df['volume'].mean()
            std_volume = df['volume'].std()
            nb_transactions = len(df)

            properties = [mean_price, 
                          std_price, 
                          mean_return, 
                          std_return, 
                          volume, 
                          mean_volume, 
                          std_volume, 
                          nb_transactions]

            result[pair] = properties
        
        return pd.DataFrame(result, index=["Mean price", 
                                            "Price standard deviation", 
                                            "Mean return", 
                                            "Return standard deviation", 
                                            "Volume", 
                                            "Mean volume", 
                                            "Standard deviation of volume", 
                                            "Number of transactions"]).T

def localization_predictable_intervals(data_manager, pair, block_size=None, window=None, test='Entropy Bias'):
    n = len(data_manager.datasets[pair])
    s = len(data_manager.symbols.keys())
    block_size = math.floor(0.5*math.log(n,s)) if block_size is None else block_size
    if window is None or window > np.floor(n - block_size + 1)/1000:
        window = int(np.floor((n - block_size + 1)/1000))
    m = int(np.floor(n/window))
    result = {}
    for i in range(0, n, window):
        symbols = data_manager.datasets[pair].iloc[i:i+window,3].values
        start = data_manager.datasets[pair].index[i]
        end = data_manager.datasets[pair].index[min(i+window-1, n-1)]

        if test == 'Entropy Bias':
            blocks = pd.DataFrame(non_overlapping_blocks(symbols, block_size))
            analyser = RandomnessAnalysis(blocks, s)
            test_result = analyser.entropy_bias_test(m=m)

        elif test == 'NP Statistic':
            blocks = pd.DataFrame(overlapping_blocks(symbols, block_size))
            analyser = RandomnessAnalysis(blocks, s)
            test_result = analyser.KL_divergence_test(m=m)
        else:
            raise ValueError("Test not recognized. Choose between 'Entropy Bias' and 'NP Statistic'.")
        
        result[start] = [end,
                         test_result.iloc[0,0], 
                         test_result.iloc[3,0], 
                         test_result.iloc[4,0],
                         test_result.iloc[6,0]]
    result = pd.DataFrame(result, index=["Timestamp End", "Test Stat", "Quantile 99%", "P-value", "Hypothesis"]).T
    result.index.name = "Timestamp Start"
    result.reset_index(inplace=True)
    result["Rank"] = result["Test Stat"].rank(ascending=False, method="dense").astype(int)
    result = result.sort_values(by="Rank")
    result = result.set_index("Rank")

    return result

def intervals_analysis(pairs,
                     symbols,
                     block_size=None,
                     max_aggregation_level=50,
                     year=None,
                     month=None,
                     day=None,
                     test='Entropy Bias'):

    if day is None:
        nb_periods = len(month)
        ref = month
    else:
        nb_periods = len(day)
        ref = day

    result = {}
    for n in range(nb_periods):
        result[ref[n]] = []
        if day is None:
            m = month[n]
            d = None
        else:
            m = month
            d = day[n]
        agg_list = []
        for level in range(1, max_aggregation_level+1):
            message = f"{year}-{m}" if day is None else f"{year}-{m}-{d}"
            print(f"[SYSTEM] Processing " + message + f" with aggregation level {level}...")
            collector = HistoricalDataCollectorParquet(pairs, year, m, d)
            collector.collect()
            data_manager = DataManager(pairs, 
                                    symbols,
                                    year=year,
                                    month=m,
                                    day=d,
                                    aggregation_level=level)
            frac_pred = []
            for pair in pairs:
                print(f"[SYSTEM] Processing {pair.upper()}...")
                df = data_manager.datasets[pair]
                n_ = len(df)
                block_size = math.floor(0.5*math.log(n_,len(symbols.keys()))) if block_size is None else block_size
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index, unit='ms')
                if day is None:
                    freq = '1D'  
                else:
                    freq = '1H'  

                grouped = df.groupby(pd.Grouper(freq=freq))
                nb_sub_periods = len(grouped)
                frac_pred.append(0)
                for interval, group in grouped:
                    if test == 'Entropy Bias':
                        blocks = pd.DataFrame(non_overlapping_blocks(group["symbol"].values, block_size))
                        analyser = RandomnessAnalysis(blocks, len(symbols.keys()))
                        test_result = analyser.entropy_bias_test()
                    elif test == 'NP Statistic':
                        blocks = pd.DataFrame(overlapping_blocks(group["symbol"].values, block_size))
                        analyser = RandomnessAnalysis(blocks, len(symbols.keys()))
                        test_result = analyser.KL_divergence_test()
                    else:
                        raise ValueError("Test not recognized. Choose between 'Entropy Bias' and 'NP Statistic'.")
                    
                    if test_result.iloc[6,0]:
                        frac_pred[-1] += 1 / nb_sub_periods
            agg_list.append(frac_pred)

        result[ref[n]] = agg_list
    
    sub_plots_comp(result,pairs, year, month, day, test)