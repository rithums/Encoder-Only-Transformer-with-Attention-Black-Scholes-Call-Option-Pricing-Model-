import numpy as np
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt


class finalDataset_series():

    #Given maturity and strike price, create dictionary where
    #Key: Day
    #Value: Data point from that day with the inputted maturity and strike price
    def gen_series(self, maturity, strike): 
        ret = {}
        for k, v in self.series_dict.items():

            #Get indices of data points with correct maturity and strike price
            match_maturity = v[:,3] == maturity
            match_strike = v[:,2] == strike
            matching_inds = np.arange(len(v))[(match_strike * match_maturity)]

            #If non-zero data points, add to dictionary
            if len(matching_inds) > 0:
                matching_rows = v[matching_inds]
                #Arbitrarily pick first row so only have 1 data point per day
                ret[k] = np.unique(matching_rows,axis=0)[0]
        return ret
    
    #Collect all combinations of maturities and strike prices from across the dataset, and create
    #the corresponding series for each of them
    def gen_all_series(self):

        #Get all maturity and strike price combinations
        all_maturity_strike_pairs = self.df[['time_to_maturity', 'strike']].to_numpy().astype('float32')
        maturity_strike_pairs = np.unique(all_maturity_strike_pairs, axis=0)


        all_series = []
        for maturity, strike in tqdm(maturity_strike_pairs):
            #Generate series
            all_series.append(self.gen_series(maturity, strike))
        return all_series

    def __init__(self):
        
        #ONLY RUN IF data/processed_series.pkl not present
        #self.preprocess_data()

        #Load preprocessed list of series
        processed_data = open("data/processed_series.pkl", "rb")
        all_series = pickle.load(processed_data)

        self.all_series_reformatted = []
        

        #Reformat each series, so instead of each item being {k : v} it is now an array [k] + v, where k is the day
        #Pick series with length at least 2 so can have input (every item except last) and target (last item in series)
        for s in all_series:
            if len(s) >= 2:
                new_series = []
                for k, v in s.items():
                    #Make the day a part of the item
                    new_series.append([k] + list(v))
                self.all_series_reformatted.append(torch.tensor(new_series, dtype=torch.float32))
        
        num_rows = 17
        data = torch.zeros((len(self.all_series_reformatted), num_rows, 5))
        masks = torch.zeros((len(self.all_series_reformatted), num_rows))
        target_inputs = torch.zeros((len(self.all_series_reformatted), 4))
        target_prices = torch.zeros((len(self.all_series_reformatted), 1))

        for i in range(len(self.all_series_reformatted)):

            input_data = self.all_series_reformatted[i][:-1]
            input_data = input_data[:, :-2]
            input_data_padded = torch.zeros((num_rows, input_data.shape[1]))
            input_data_padded[:input_data.shape[0]] = input_data

            data[i] = input_data_padded

            mask = torch.zeros((num_rows))
            mask[:input_data.shape[0]] = 1

            masks[i] = mask


            input_to_target_call_price = self.all_series_reformatted[i][-1][1:-2]
            target_inputs[i] = input_to_target_call_price

            target_call_price = self.all_series_reformatted[i][-1][-1].view(-1)
            target_prices[i] = target_call_price



        self.final_input = data
        self.final_masks = masks
        self.final_target_inputs = target_inputs
        self.final_target_prices = target_prices

    def analyze(self):
        print("Num series: " + str(len(self.final_target_inputs)))

        series_lengths = {}
        for i in range(17):
            series_lengths[i] = 0
        for mask in self.final_masks:
            series_length = torch.sum(mask).item()
            series_lengths[series_length] += 1

        plt.plot(list(series_lengths.keys()), list(series_lengths.values()))
        plt.title('Lengths of Series in Dataset')
        plt.xlabel('Number of contracts per series')
        plt.ylabel('Count')
        plt.savefig("figures/series_counts.png")


dataset = finalDataset_series()
dataset.analyze()
            
            

        


            

    
    

