import pandas as pd
import numpy as np
import os
import math
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.BlockConstructors import non_overlapping_blocks, overlapping_blocks
from data.DataCollectors import RAW_DATA_FOLDER

RAW_PARQUET_FOLDER = "data/raw_data_parquet"

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
            parquet_file = os.path.join(RAW_PARQUET_FOLDER, f"{pair.upper()}-trades-{self.when}.parquet")
            csv_file = os.path.join(RAW_DATA_FOLDER, f"{pair.upper()}-trades-{self.when}.csv")

            if os.path.exists(parquet_file):
                print(f"[DataManager] Loading parquet file: {parquet_file}")
                df = pd.read_parquet(parquet_file)
                    # Si colonnes sont anonymes (cas rare), appliquer fix
                if set(df.columns) == set(range(df.shape[1])):
                    df.columns = ["trade_id", "price", "volume", "quote_qty",
                                "timestamp", "is_buyer_maker", "is_best_match"]

            elif os.path.exists(csv_file):
                print(f"[DataManager] Loading csv file: {csv_file}")
                df = pd.read_csv(csv_file,
                                header=None,
                                names=["trade_id", "price", "volume", "quote_qty",
                                        "timestamp", "is_buyer_maker", "is_best_match"])
            else:
                raise FileNotFoundError(f"Neither parquet nor csv file found for {pair} ({self.when})")

            df["timestamp"] = pd.to_numeric(df["timestamp"])
            ts = df["timestamp"].astype(float)
            max_ts = ts.max()

            if max_ts > 1e14:
                unit = "us"
            elif max_ts > 1e11:
                unit = "ms"
            else:
                unit = "s"

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit)
            df['log price'] = np.log(df['price'])
            df.sort_index(inplace=True)

            if self.aggregate_by_time:
                df = df.groupby('timestamp').agg({
                    'timestamp': 'first',
                    'volume': 'sum',
                    'log price': 'last'
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
            dataset["symbol"] = dataset["symbol"].astype(np.int8)

            self.datasets[pair] = dataset

    def save_data(self):
        for i, (pair, dataset) in enumerate(self.datasets.items()):
            if self.checks[i]:
                continue
            filename = os.path.join(
                PREPROCCESSED_DATA_FOLDER,
                f"{pair.upper()}_processed"
                + (f"_{self.when}" if self.when is not None else "_REAL_TIME")
                + f"_S={len(self.symbols.keys())}"
                + f"_A={self.aggregation_level}.parquet"  
            )
            dataset.to_parquet(filename, index=True) 

    
    def exist(self, pair, block_size=None):
        exist = False
        if block_size is None:
            filename = os.path.join(
                PREPROCCESSED_DATA_FOLDER,
                f"{pair.upper()}_processed"
                + (f"_{self.when}" if self.when is not None else "_REAL_TIME")
                + f"_S={len(self.symbols.keys())}"
                + f"_A={self.aggregation_level}.parquet"  
            )
            if os.path.exists(filename):
                self.datasets[pair] = pd.read_parquet(filename) 
                exist = True
        else:
            filename = (
                f"{BLOCKS_FOLDER}/{pair.upper()}"
                + (f"_{self.when}" if self.when is not None else "_REAL_TIME")
                + f"_A={self.aggregation_level}"
                + f"_size={block_size}.csv"
            )
            if os.path.exists(filename):
                exist = True
        return exist

    def block_constructor(self, pairs=None, block_size=None, overlapping=False, save=True):
        """
        Crée les blocs pour chaque pair.
        Sauvegarde maintenant les blocs au format parquet (plus rapide que CSV).
        """
        if pairs is None:
            pairs = self.assets_pairs
        self.blocks = {}
        status = "F" if not overlapping else "T"
        
        for pair in pairs:
            filename = (
                f"{BLOCKS_FOLDER}/{pair.upper()}"
                + (f"_{self.when}" if self.when is not None else "_REAL_TIME")
                + f"_A={self.aggregation_level}"
                + f"_O={status}"
                + f"_size={block_size}.parquet"
            )

            # Charger si déjà existant
            if os.path.exists(filename):
                self.blocks[pair] = pd.read_parquet(filename)
                continue

            dataset = self.datasets[pair]
            n = len(dataset)
            s = len(self.symbols.keys())
            symbols = dataset['symbol'].values.astype(np.int8)

            block_size_local = math.floor(0.5 * math.log(n, s)) if block_size is None else block_size

            if len(symbols) < block_size_local:
                print(f"[WARNING] Not enough data to construct blocks for {pair}. Skipping.")
                continue

            # Construction rapide des blocs
            if overlapping:
                shape = (symbols.shape[0] - block_size_local + 1, block_size_local)
                strides = (symbols.strides[0], symbols.strides[0])
                blocks = np.lib.stride_tricks.as_strided(symbols, shape=shape, strides=strides)
            else:
                n_blocks = symbols.shape[0] // block_size_local
                trimmed_symbols = symbols[:n_blocks * block_size_local]
                blocks = trimmed_symbols.reshape(-1, block_size_local)

            blocks_df = pd.DataFrame(blocks)
            self.blocks[pair] = blocks_df

            if save:
                blocks_df.to_parquet(filename, index=False)  # <-- Parquet !
        
        return self.blocks

    
    def matching_aggregation_for(self, asset_other, blocks, year, month, day=None):
        self_asset = self.assets_pairs[0]
        n_self_blocks = len(self.datasets[self_asset]['symbol'])
        M = blocks or n_self_blocks

        dm0 = DataManager([asset_other],
                        self.symbols,
                        year, month, day,
                        aggregation_level=1)

        n_other = len(dm0.datasets[asset_other]['symbol'])

        # Trouve l’agrégation qui réduit n_other à environ M
        return max(1, n_other // M)
    
    def compute_tx_frequency_metrics(
        self,
        pair: str,
        max_interval_sec: float = 1.0
    ) -> dict:

        df = self.datasets[pair]
        ts = df.index.to_series().sort_index()
        dt = ts.diff().dt.total_seconds().iloc[1:]

        total_tx = len(ts)
        avg_dt = dt.mean() if not dt.empty else float('nan')
        frac_within = (dt <= max_interval_sec).sum() / len(dt) if not dt.empty else 0.0

        return {
            'aggregation_level': self.aggregation_level,
            'total_transactions': total_tx,
            'avg_interval_sec': avg_dt,
            'fraction_within_1s': frac_within
        }


    @staticmethod
    def summarize_transaction_characteristics(
        pairs: list[str],
        symbols: dict,
        year: int,
        month: int,
        day: int = None,
        aggregation_levels: list[int] = [1, 5, 10, 50],
        max_interval_sec: float = 1.0
    ) -> pd.DataFrame:
        
        records = []
        for pair in pairs:
            for lvl in aggregation_levels:
                dm = DataManager(
                    asset_pairs=[pair],
                    symbols=symbols,
                    year=year, month=month, day=day,
                    aggregation_level=lvl
                )
                dm.load_data()
                dm.checks = [False] * len(dm.checks)   # force re-preprocess
                dm.preprocess_data()

                m = dm.compute_tx_frequency_metrics(pair, max_interval_sec)
                m['pair'] = pair
                records.append(m)

        df = pd.DataFrame(records)
        return df.set_index(['pair','aggregation_level'])

    def plot_transaction_frequency_two(
        pairs: list[str],
        symbols: dict,
        year: int,
        month: int,
        day: int,
        aggregation_levels: list[int],
        max_interval_sec: float = 1.0
    ) -> go.Figure:

        fig = go.Figure()
        for pair in pairs:
            df = DataManager.summarize_transaction_characteristics(
                pairs=pair,
                symbols=symbols,
                year=year,
                month=month,
                day=day,
                aggregation_levels=aggregation_levels,
                max_interval_sec=max_interval_sec
            )
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['fraction_within_1s'],
                mode='lines+markers',
                name=pair
            ))

        fig.update_layout(
            title=f"Fraction of intervals ≤ {max_interval_sec}s",
            xaxis_title="Aggregation Level",
            yaxis_title=f"Fraction ≤ {max_interval_sec}s",
            template="plotly_white",
            height=500,
            width=800
        )
        return fig
    
    @staticmethod
    def plot_transaction_counts_express(
        pairs: list[str],
        symbols: dict,
        year: int,
        month: int,
        day: int = None,
        aggregation_levels: list[int] = []
    ) -> go.Figure:

        records = []

        for pair in pairs:
            for lvl in aggregation_levels:
                dm = DataManager(
                    asset_pairs=[pair],
                    symbols=symbols,
                    year=year, month=month, day=day,
                    aggregation_level=lvl
                )
                # important : on veut forcer un rechargement propre
                dm.preprocess_data()
                m = dm.compute_tx_frequency_metrics(pair)
                m["pair"] = pair
                m["aggregation_level"] = lvl
                records.append(m)

        df = pd.DataFrame(records)

        fig = px.line(
            df,
            x="aggregation_level",
            y="total_transactions",
            color="pair",
            markers=True,
            title="Total Transactions vs Aggregation Level"
        )

        fig.update_layout(
            xaxis_title="Aggregation Level",
            yaxis_title="Total Transactions",
            template="plotly_white"
        )

        return fig

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
    
    