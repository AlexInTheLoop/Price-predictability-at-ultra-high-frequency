import numpy as np
import pandas as pd
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
            filename = os.path.join(PREPROCCESSED_DATA_FOLDER, f"{pair.upper()}_processed" + 
                                    (f"_{when}" if when is not None else "_REAL_TIME") +
                                    f"_S={s}" + 
                                    f"_A={aggregation_level}.csv")
            if not os.path.exists(filename):
              raise FileNotFoundError(f"File {filename} does not exist. Launch the data collector/manager first.")
            df = pd.read_csv(filename, index_col="timestamp", parse_dates=["timestamp"])

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

