import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.BlockConstructors import non_overlapping_blocks, overlapping_blocks
from main.RandomnessAnalysis import RandomnessAnalysis
from utils.VisualizationTools import plot_predictability

DEFAULT_SYMBOLS = {
                    0: [(-np.inf, 0), (False, False)],
                    1: [(0, np.inf), (False, False)]
                }

class DataGenerator:
    def __init__(self,symbols=DEFAULT_SYMBOLS):
        self.s = len(symbols.keys())
    
    def lambda_model(self,
                     max_aggration_level=50,
                     series_length=10**5,
                     nb_initial_orders=21,
                     n_days=80,
                     alpha=1.63,
                     prob=0.38,
                     test = 'Entropy Bias',
                     overlapping=False,
                     k=None):
        self.freq_pred = np.zeros(max_aggration_level)
        for day in range(n_days):
            nb_orders = nb_initial_orders
            signs = np.random.binomial(n=1,p=0.5,size=nb_orders)
            volumes = np.ceil(np.random.pareto(a=alpha,size=nb_orders))
            output = np.zeros(series_length)
            for t in range(series_length):
                new_volume = np.random.binomial(n=1,p=prob)
                if nb_orders == 0 or new_volume == 1:
                    volumes = np.append(volumes,np.ceil(np.random.pareto(a=alpha,size=1)))
                    signs = np.append(signs,np.random.binomial(n=1,p=0.5,size=1))
                    nb_orders += 1
                random_order = np.random.randint(0,nb_orders)
                output[t] = signs[random_order]
                volumes[random_order] -= 1
                if int(volumes[random_order]) == 0:
                    volumes = np.delete(volumes,int(random_order))
                    signs = np.delete(signs, random_order)
                    nb_orders -= 1
            for level in range(1,max_aggration_level+1):
                data = output[0:np.size(output):level]
                k = int(np.round(np.log(np.size(data))/np.log(self.s)/2)) if k is None else k
                if overlapping:
                    blocks_df = pd.DataFrame(overlapping_blocks(symbols=data, block_size=k))
                else:
                    blocks_df = pd.DataFrame(non_overlapping_blocks(symbols=data, block_size=k))
                analysis = RandomnessAnalysis(blocks_df, self.s, k)
                if test == 'Entropy Bias':
                    result = analysis.entropy_bias_test()
                elif test == 'KL Divergence':
                    result = analysis.KL_divergence_test()
                else:
                    raise ValueError("Test not implemented")
                if result.iloc[6,0]:
                    self.freq_pred[level-1] += 1
        self.freq_pred = self.freq_pred / n_days
        plot_predictability(range(1,max_aggration_level+1),self.freq_pred)

if __name__ == "__main__":
    data_gen = DataGenerator()
    data_gen.lambda_model(test='KL Divergence',overlapping=True)
