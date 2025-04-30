import numpy as np
import pandas as pd
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.BlockConstructors import non_overlapping_blocks, overlapping_blocks
from main.RandomnessAnalysis import RandomnessAnalysis
from utils.VisualizationTools import plot_predictability, plot_all_models

DEFAULT_SYMBOLS = {
                    0: [(-np.inf, 0), (False, False)],
                    1: [(0, np.inf), (False, False)]
                }

class DataGenerator:
    def __init__(self,symbols=DEFAULT_SYMBOLS):
        self.s = len(symbols.keys())
        self.freq_pred_agg = {}
        self.freq_pred_timelag = {}
    
    def lambda_model(self,
                     max_aggregation_level=50,
                     max_time_lag=50,
                     series_length=10**5,
                     nb_initial_orders=21,
                     n_days=80,
                     alpha=1.63,
                     prob=0.38,
                     test = 'KL Divergence',
                     overlapping=True,
                     k=None,
                     plots=(True,False)):
        self.freq_pred_agg['λ-model'] = np.zeros(max_aggregation_level)
        self.freq_pred_timelag['λ-model'] = np.zeros(max_time_lag)
        for day in range(n_days):
            print(f"Day {day+1}/{n_days}")
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
            for level in range(1,max_aggregation_level+1):
                data = output[0:np.size(output):level]
                k = int(np.round(np.log(np.size(data))/np.log(self.s)/2)) if k is None else k
                if overlapping:
                    blocks_df = pd.DataFrame(overlapping_blocks(symbols=data, block_size=k))
                else:
                    blocks_df = pd.DataFrame(non_overlapping_blocks(symbols=data, block_size=k))
                analysis = RandomnessAnalysis(blocks_df, self.s)
                if test == 'Entropy Bias':
                    result = analysis.entropy_bias_test()
                elif test == 'KL Divergence':
                    result = analysis.KL_divergence_test()
                else:
                    raise ValueError("Test not implemented")
                if result.iloc[6,0]:
                    self.freq_pred_agg['λ-model'][level-1] += 1
            for time_lag in range(1,max_time_lag+1):
                blocks_df = pd.DataFrame(self._build_blocks_by_timelag(output, time_lag))
                k = int(np.round(np.log(np.size(data))/np.log(self.s)/2)) if k is None else k
                analysis = RandomnessAnalysis(blocks_df, self.s)
                if test == 'Entropy Bias':
                    result = analysis.entropy_bias_test()
                elif test == 'KL Divergence':
                    result = analysis.KL_divergence_test()
                else:
                    raise ValueError("Test not implemented")
                if result.iloc[6,0]:
                    self.freq_pred_timelag['λ-model'][time_lag-1] += 1
        self.freq_pred_agg['λ-model'] = self.freq_pred_agg['λ-model'] / n_days
        if plots[0]:
            plot_predictability(list(range(1,max_aggregation_level+1)),self.freq_pred_agg['λ-model'])
        if plots[1]:
            plot_predictability(list(range(1,max_time_lag+1)),self.freq_pred_timelag['λ-model'],x_label='time lag')
    
    def OD_model(self,
                 max_aggregation_level=50,
                 max_time_lag=50,
                 series_length=10**5,                   # Number of executed orders to collect
                 nb_traders=1000,                       # Number of traders
                 n_days=80,                             # Number of days to simulate
                 max_memory=100,                        # Maximum memory used by traders to estimate the trend
                 tau = 2*100,                           # Maximum time to wait before canceling an order
                 pf = 1000,                             # Fundamental price
                 sigma1 = 1,                            # Standard deviation of the sensitivity of the traders to fundamental gap
                 sigma2 = 1.4,                          # Standard deviation of the sensitivity of the traders to the local trend
                 n0 = 0.5,                              # Standard deviation of the noise      
                 lamba_ = 0.5,                          # Interaction probability
                 k_max = 0.5,                           # Agressiveness probability
                 delta = 0.1,                           # Tick size
                 test = 'Entropy Bias',
                 overlapping=False,
                 k=None,
                 plots=(True,False)):
        self.freq_pred_agg['OD-model'] = np.zeros(max_aggregation_level)
        self.freq_pred_timelag['OD-model'] = np.zeros(max_time_lag)
        day = 0
        while day < n_days:
            while True:
                try:
                    day += 1
                    print(f"Day {day}/{n_days}")
                    g1 = abs(np.random.normal(0, sigma1, nb_traders))
                    g2 = np.random.normal(0, sigma2, nb_traders)
                    noise = np.random.normal(0, n0, nb_traders)
                    ki = np.random.uniform(0, k_max, nb_traders)
                    L = [random.randint(1, max_memory) for _ in range(nb_traders)]
                    bid, ask, time_of_bid, time_of_ask = [], [], [], []
                    p = pf * np.ones(2)
                    output_prices, output_signs = [], []
                    t = 1
                    while np.size(output_prices) < series_length:
                        t += 1
                        if np.size(time_of_bid) > 0:
                            time_of_bid += 1
                            if time_of_bid[0] > tau:
                                bid = np.delete(bid, 0)
                                time_of_bid = np.delete(time_of_bid, 0)
                        if np.size(time_of_ask) > 0:
                            time_of_ask += 1
                            if time_of_ask[0] > tau:
                                ask = np.delete(ask, 0)
                                time_of_ask = np.delete(time_of_ask, 0)
                        liquidity_check = np.random.binomial(size=1, n=1, p=lamba_)
                        if liquidity_check == 0:
                            if np.size(bid) * np.size(ask) > 0:
                                p = np.append(p, (np.max(bid) + np.min(ask)) / 2)
                            else:
                                p = np.append(p, p[t-1])
                        else:
                            trader_nb = random.randrange(nb_traders)
                            L_local = np.min([L[trader_nb], t-1])
                            if np.any(p[(t-L_local-1):(t-1)] == 0):
                                r_L = 0
                            else:
                                r_L = sum(np.divide(
                                    p[(t-L_local):t] - p[(t-L_local-1):(t-1)],
                                    p[(t-L_local-1):(t-1)]
                                )) / L_local
                            eps = np.random.normal(0, 1)
                            r_hat = g1[trader_nb] * (pf - p[t-1]) / p[t-1] + g2[trader_nb] * r_L + noise[trader_nb] * eps
                            p_hat = p[t-1] * np.exp(r_hat)
                            if p_hat > p[t-1]:
                                bid_t = self._roundbyticksize(p_hat * (1 - ki[trader_nb]), delta)
                                if np.size(ask) > 0 and bid_t >= np.min(ask):
                                    p = np.append(p, np.min(ask))
                                    output_signs = np.append(output_signs, 1)
                                    output_prices = np.append(output_prices, np.min(ask))
                                    time_of_ask = np.delete(time_of_ask, np.argmin(ask))
                                    ask = np.delete(ask, np.argmin(ask))
                                else:
                                    bid = np.append(bid, bid_t)
                                    time_of_bid = np.append(time_of_bid, 0)
                                    if np.size(bid) * np.size(ask) > 0:
                                        p = np.append(p, (np.max(bid) + np.min(ask)) / 2)
                                    else:
                                        p = np.append(p, p[t-1])
                            else:
                                ask_t = self._roundbyticksize(p_hat * (1 + ki[trader_nb]), delta)
                                if np.size(bid) > 0 and ask_t <= np.max(bid):
                                    p = np.append(p, np.max(bid))
                                    output_signs = np.append(output_signs, 0)
                                    output_prices = np.append(output_prices, np.max(bid))
                                    time_of_bid = np.delete(time_of_bid, np.argmax(bid))
                                    bid = np.delete(bid, np.argmax(bid))
                                else:
                                    ask = np.append(ask, ask_t)
                                    time_of_ask = np.append(time_of_ask, 0)
                                    if np.size(bid) * np.size(ask) > 0:
                                        p = np.append(p, (np.max(bid) + np.min(ask)) / 2)
                                    else:
                                        p = np.append(p, p[t-1])
                            if np.isnan(r_hat) or t > 10 * series_length:
                                print(t, np.size(output_prices))
                                print(p)
                                print(p_hat, r_hat, (pf - p[t-1]) / p[t-1], r_L)
                                print(bid_t, ask_t, ki[trader_nb])
                                print(bid)
                                print(ask)
                                raise Exception("error in price dynamics")
                    for level in range(1, max_aggregation_level+1):
                        prices = output_prices[range(0, np.size(output_prices), level)]
                        returns = prices[1:] - prices[:-1]
                        returns = np.delete(returns, np.where(returns == 0)[0])
                        data = np.zeros_like(returns)
                        data[np.where(returns > 0)[0]] = 1
                        data[np.where(returns < 0)[0]] = 0
                        k = int(np.round(np.log(np.size(data)) / np.log(self.s) / 2)) if k is None else k
                        if overlapping:
                            blocks_df = pd.DataFrame(overlapping_blocks(symbols=data, block_size=k))
                        else:
                            blocks_df = pd.DataFrame(non_overlapping_blocks(symbols=data, block_size=k))
                        analysis = RandomnessAnalysis(blocks_df, self.s)
                        if test == 'Entropy Bias':
                            result = analysis.entropy_bias_test()
                        elif test == 'KL Divergence':
                            result = analysis.KL_divergence_test()
                        else:
                            raise ValueError("Test not implemented")
                        if result.iloc[6, 0]:
                            self.freq_pred_agg['OD-model'][level-1] += 1
                    
                    for time_lag in range(1, max_time_lag+1):
                        blocks_df = pd.DataFrame(self._build_blocks_by_timelag(output_prices, time_lag))
                        k = int(np.round(np.log(np.size(data)) / np.log(self.s) / 2)) if k is None else k
                        analysis = RandomnessAnalysis(blocks_df, self.s)
                        if test == 'Entropy Bias':
                            result = analysis.entropy_bias_test()
                        elif test == 'KL Divergence':
                            result = analysis.KL_divergence_test()
                        else:
                            raise ValueError("Test not implemented")
                        if result.iloc[6, 0]:
                            self.freq_pred_timelag['OD-model'][time_lag-1] += 1
                    break
                except Exception as e:
                    print(f"Error on day {day}: {e}")
                    day -= 1
        self.freq_pred_agg['OD-model'] = self.freq_pred_agg['OD-model'] / n_days
        if plots[0]:
            plot_predictability(list(range(1, max_aggregation_level+1)), self.freq_pred_agg['OD-model'])
        if plots[1]:
            plot_predictability(list(range(1, max_time_lag+1)), self.freq_pred_timelag['OD-model'],x_label='time lag')

    @staticmethod
    def _roundbyticksize(x,delta):
        return np.round(x/delta)*delta
    
    def TS_model(self,
                 max_aggregation_level=50,
                 max_time_lag=50,
                 beta=0.42,
                 l0=20,
                 gamma0=2.8*10**(-3),
                 n_days=80,
                 series_length=10**5,
                 sigma1=0.01,
                 sigma2=0.01,
                 p0=0,
                 test='Entropy Bias',
                 overlapping=False,
                 k=None,
                 plots=(True,False)):
        G0=self._G0_function(np.arange(series_length),gamma0,l0,beta)
        self.freq_pred_agg['TS-model'] = np.zeros(max_aggregation_level)
        self.freq_pred_timelag['TS-model'] = np.zeros(max_time_lag)
        for day in range(n_days):
            print(f"Day {day+1}/{n_days}")
            eps=np.random.normal(0,sigma1,series_length)
            eta=np.random.normal(0,sigma2,series_length)
            lnV = np.random.normal(5.5,1.8,series_length)
            signs=np.zeros(series_length)
            price=np.zeros(series_length)
            price[0]=p0+eps[0]
            if price[0]>p0:
                signs[0]=1
            else:
                signs[0]=0
            for t in range(1,series_length):
                price[t]=np.dot(np.multiply(G0[0:t],signs[0:t]),lnV[0:t])+sum(eta[0:t])+eps[t]
            for level in range(1, max_aggregation_level+1):
                        prices = price[range(0, np.size(price), level)]
                        returns = prices[1:] - prices[:-1]
                        returns = np.delete(returns, np.where(returns == 0)[0])
                        data = np.zeros_like(returns)
                        data[np.where(returns > 0)[0]] = 1
                        data[np.where(returns < 0)[0]] = 0
                        k = int(np.round(np.log(np.size(data)) / np.log(self.s) / 2)) if k is None else k
                        if overlapping:
                            blocks_df = pd.DataFrame(overlapping_blocks(symbols=data, block_size=k))
                        else:
                            blocks_df = pd.DataFrame(non_overlapping_blocks(symbols=data, block_size=k))
                        analysis = RandomnessAnalysis(blocks_df, self.s)
                        if test == 'Entropy Bias':
                            result = analysis.entropy_bias_test()
                        elif test == 'KL Divergence':
                            result = analysis.KL_divergence_test()
                        else:
                            raise ValueError("Test not implemented")
                        if result.iloc[6, 0]:
                            self.freq_pred_agg['TS-model'][level-1] += 1

            for time_lag in range(1, max_time_lag+1):
                blocks_df = pd.DataFrame(self._build_blocks_by_timelag(price, time_lag))
                k = int(np.round(np.log(np.size(data)) / np.log(self.s) / 2)) if k is None else k
                analysis = RandomnessAnalysis(blocks_df, self.s)
                if test == 'Entropy Bias':
                    result = analysis.entropy_bias_test()
                elif test == 'KL Divergence':
                    result = analysis.KL_divergence_test()
                else:
                    raise ValueError("Test not implemented")
                if result.iloc[6, 0]:
                    self.freq_pred_timelag['TS-model'][time_lag-1] += 1
            
        self.freq_pred_agg['TS-model'] = self.freq_pred_agg['TS-model'] / n_days
        if plots[0]:
            plot_predictability(list(range(1, max_aggregation_level+1)), self.freq_pred_agg['TS-model'])
        if plots[1]:
            plot_predictability(list(range(1, max_time_lag+1)), self.freq_pred_timelag['TS-model'],x_label='time lag')

    @staticmethod
    def _G0_function(l,l0=20,gamma0=2.8*10**(-3),beta=0.42):
        return gamma0 / (l+l0)**beta
    
    @staticmethod
    def _build_blocks_by_timelag(prices, time_lag):
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.log(np.divide(prices[1:], prices[:-1], where=prices[:-1] != 0))
            returns[np.isnan(returns)] = 0
        symbols = (returns > 0).astype(int)
        blocks = np.stack([symbols[:-time_lag], symbols[time_lag:]], axis=1)
        return blocks

    def plot_all(self,what='aggregation level',max_aggregation_level=50,max_time_lag=50):
        models = ['λ-model','OD-model','TS-model']
        if what == 'aggregation level':
            all_present = all(item in self.freq_pred_agg for item in models)
            if not(all_present):
                self.lambda_model(max_aggregation_level=max_aggregation_level,
                                  max_time_lag=max_time_lag,
                                  plots=(False,False))
                self.OD_model(max_aggregation_level=max_aggregation_level,
                              max_time_lag=max_time_lag,
                              plots=(False,False))
                self.TS_model(max_aggregation_level=max_aggregation_level,
                              max_time_lag=max_time_lag,
                              plots=(False,False))
            y = {'λ-model': self.freq_pred_agg['λ-model'],
                 'OD-model': self.freq_pred_agg['OD-model'],
                 'TS-model': self.freq_pred_agg['TS-model']}
            plot_all_models(list(range(1, max_aggregation_level+1)),y)
        elif what == 'time lag':
            all_present = all(item in self.freq_pred_timelag for item in models)
            if not(all_present):
                self.lambda_model(max_aggregation_level=max_aggregation_level,
                                  max_time_lag=max_time_lag,
                                  plots=(False,False))
                self.OD_model(max_aggregation_level=max_aggregation_level,
                              max_time_lag=max_time_lag,
                              plots=(False,False))
                self.TS_model(max_aggregation_level=max_aggregation_level,
                              max_time_lag=max_time_lag,
                              plots=(False,False))
            y = {'λ-model': self.freq_pred_timelag['λ-model'],
                 'OD-model': self.freq_pred_timelag['OD-model'],
                 'TS-model': self.freq_pred_timelag['TS-model']}
            plot_all_models(list(range(1, max_time_lag+1)),y,x_label='time lag')
        else:
            raise ValueError("Invalid value for 'what'. Choose either 'aggregation level' or 'time lag'.")
            

                        
if __name__ == "__main__":
    data_gen = DataGenerator()
    #data_gen.lambda_model(test='KL Divergence',overlapping=True,max_aggregation_level=5,n_days=2,plots=(False,True))
    #data_gen.OD_model(test='KL Divergence',overlapping=True,max_aggregation_level=5,n_days=2,plots=(False,True))
    #data_gen.TS_model(test='KL Divergence',overlapping=True,max_aggregation_level=10,n_days=2,plots=(False,True))
    data_gen.plot_all(what='aggregation level')
    data_gen.plot_all(what='time lag')
    