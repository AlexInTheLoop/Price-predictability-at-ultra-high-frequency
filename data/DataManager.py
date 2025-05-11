import pandas as pd
import numpy as np
import os
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.BlockConstructors import non_overlapping_blocks, overlapping_blocks
from data.DataCollectors import RAW_DATA_FOLDER

PREPROCCESSED_DATA_FOLDER = "data/preprocessed_data"
if not os.path.exists(PREPROCCESSED_DATA_FOLDER):
    os.makedirs(PREPROCCESSED_DATA_FOLDER)

BLOCKS_FOLDER = "data/blocks"
if not os.path.exists(BLOCKS_FOLDER):
    os.makedirs(BLOCKS_FOLDER)

symbols_example = {
    0: [(-np.inf, -0.00001), (False, False)],
    1: [(-0.00001, 0.00001), (True, False)],
    2: [(0.00001, np.inf), (True, True)]
}

class DataManager:
    def __init__(self, 
                 asset_pairs, 
                 symbols, 
                 year,
                 month,
                 day=None,
                 aggregation_level = 1, 
                 exclude_zero=True,
                 aggregate_by_time=False):
        self.assets_pairs = asset_pairs
        self.symbols = symbols
        if year is None or month is None:
            self.when = None
        elif day is None:
            self.when = f"{year}-" + f"{int(month):02d}"
        else:
            self.when = f"{year}-" + f"{int(month):02d}-" + f"{int(day):02d}"
        self.exclude_zero = exclude_zero
        self.aggregation_level = aggregation_level
        self.aggregate_by_time = aggregate_by_time
        self.datasets = {}
        self.__post_init__()
    
    def __post_init__(self):
        self.load_data()
        self.checks = []
        for pair in self.assets_pairs:
            self.checks.append(self.exist(pair))
        self.preprocess_data()
        self.save_data()

    def load_data(self):
        for pair in self.assets_pairs:
            if self.when is None:
                filename = f"{RAW_DATA_FOLDER}/REAL_TIME_{pair.upper()}.csv"
                df = pd.read_csv(filename, usecols=["timestamp", "price"])
            else:
                filename = os.path.join(RAW_DATA_FOLDER, f"{pair.upper()}-trades-{self.when}.csv")
                df = pd.read_csv(filename,
                                 header=None, 
                                 names=["trade_id", "price", "volume", "quote_qty","timestamp","is_buyer_maker","is_best_match"])
            df["timestamp"] = pd.to_numeric(df["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            
            #df['duration'] = df['timestamp'].diff().fillna(0)
            df['log price'] = np.log(df['price'])
            df.sort_index(inplace=True)
            if self.aggregate_by_time:
                df = df.groupby('timestamp').agg({
                    'timestamp':'first',
                    'volume':'sum',
                    'log price':'last'#,
                    #'duration':'sum'
                }).reset_index(drop=True)
            df.set_index("timestamp", inplace=True)
            self.datasets[pair] = df

    def preprocess_data(self):
        for i, (pair,dataset) in enumerate(self.datasets.items()):
            if self.checks[i]:
                continue
            dataset = dataset.copy()
            dataset.reset_index(inplace=True)
            dataset['key'] = np.floor(np.arange(len(dataset)) / self.aggregation_level).astype(int)
            dataset = dataset.groupby('key').agg({
                'timestamp': 'last',
                'log price': 'mean',
                'volume': 'sum'#,
                #'duration': 'sum'
            }).rename(columns={'log price':'price'}).reset_index(drop=True)
            #.rename(columns={'log price':'mean','duration':'duration'}).reset_index(drop=True)
            dataset["returns"] = dataset["price"].diff().fillna(0)
            dataset.set_index("timestamp", inplace=True)
            dataset.dropna(inplace=True)

            if self.exclude_zero:
                dataset = dataset[dataset["returns"] != 0].copy()

            bins = []
            labels = []
            for label, (bounds, inclusions) in self.symbols.items():
                lower, upper = bounds
                lower_inclusive, upper_inclusive = inclusions

                if not lower_inclusive and lower != -np.inf:
                    lower += 1e-10
                if not upper_inclusive and upper != np.inf:
                    upper -= 1e-10

                bins.append((lower, upper))
                labels.append(label)

            flat_bins = [b[0] for b in bins] + [bins[-1][1]]

            if not all(flat_bins[i] < flat_bins[i + 1] for i in range(len(flat_bins) - 1)):
                raise ValueError("Les bornes des bins ne sont pas strictement croissantes : ", flat_bins)


            dataset["symbol"] = pd.cut(
                                        dataset["returns"],
                                        bins=flat_bins,
                                        labels=labels,
                                        include_lowest=True
                                    ) 
            
            dataset["symbol"] = dataset["symbol"].astype(int)

            self.datasets[pair] = dataset

    def save_data(self):
        for i, (pair,dataset) in enumerate(self.datasets.items()):
            if self.checks[i]:
                continue
            filename = os.path.join(PREPROCCESSED_DATA_FOLDER, f"{pair.upper()}_processed" + 
                                    (f"_{self.when}" if self.when is not None else "_REAL_TIME") +
                                    f"_S={len(self.symbols.keys())}" + 
                                    f"_A={self.aggregation_level}.csv")
            dataset.to_csv(filename, index=True)
    
    def exist(self, pair, block_size=None):
        exist = False

        if block_size is None:
            filename = os.path.join(
                PREPROCCESSED_DATA_FOLDER,
                f"{pair.upper()}_processed" +
                (f"_{self.when}" if self.when is not None else "_REAL_TIME") +
                f"_S={len(self.symbols.keys())}" +
                f"_A={self.aggregation_level}.csv"
            )
            if os.path.exists(filename):
                self.datasets[pair] = pd.read_csv(filename, index_col="timestamp", parse_dates=["timestamp"])
                exist = True
        else:
            filename = (
                f"{BLOCKS_FOLDER}/{pair.upper()}"
                + (f"_{self.when}" if self.when is not None else "_REAL_TIME")
                + f"_size={block_size}.csv"
            )

            if os.path.exists(filename):
                exist = True
        return exist

    def block_constructor(self, pairs=None, block_size=None, overlapping=False):
        if pairs is None:
            pairs = self.assets_pairs
        self.blocks = {}
        for pair in pairs:
            filename = (
                f"{BLOCKS_FOLDER}/{pair.upper()}"
                + (f"_{self.when}" if self.when is not None else "_REAL_TIME")
                + f"_size={block_size}.csv"
            )
            if self.exist(pair, block_size):
                self.blocks[pair] = pd.read_csv(filename, header=None)
                continue
            n = len(self.datasets[pair])
            s = len(self.symbols.keys())
            symbols = self.datasets[pair]['symbol'].values

            block_size = math.floor(0.5*math.log(n,s)) if block_size is None else block_size

            if overlapping:
                self.blocks[pair] = overlapping_blocks(symbols, block_size)
            else:
                self.blocks[pair] = non_overlapping_blocks(symbols, block_size)
            
            self.blocks[pair] = pd.DataFrame(self.blocks[pair])
            self.blocks[pair].to_csv(filename, index=False, header=False)

        return self.blocks


if __name__ == "__main__":
    asset_pairs = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"
    ]

    symbols = {
                0: [(-np.inf, 0), (False, False)],
                1: [(0, np.inf), (False, False)]
            }

    data_manager = DataManager(asset_pairs, symbols, 2024, 11)
    blocks = data_manager.block_constructor(block_size=3, overlapping=False)