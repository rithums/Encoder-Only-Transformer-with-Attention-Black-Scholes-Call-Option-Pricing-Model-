import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm


class finalDataset_series(Dataset):

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

    def preprocess_data(self):

        self.df = pd.read_csv("data/UnderlyingOptionsEODQuotes_2024-10.csv")

        # Calculate the target variable (mid price)
        bid_prices = self.df['bid_1545'].values
        ask_prices = self.df['ask_1545'].values
        self.target = (bid_prices + ask_prices) / 2  # Mid price as target

        # Calculate the stock price as the midpoint of underlying bid and ask prices
        self.df['stock_price'] = (self.df['underlying_bid_1545'] + self.df['underlying_ask_1545']) / 2

        # Calculate `time to maturity` in years from `expiration` and `quote_date`
        self.df['quote_date'] = pd.to_datetime(self.df['quote_date'])
        self.df['expiration'] = pd.to_datetime(self.df['expiration'])
        self.df['time_to_maturity'] = (self.df['expiration'] - self.df['quote_date']).dt.days / 365.0

        implied_volatility = 0.2  # Default value for implied volatility
        self.df['implied_volatility'] = implied_volatility


        interestRate = 0.05  # Default value for interest rate
        self.df['interestRate'] = interestRate

        self.df['callPrice'] = self.target



        self.params = self.df[['quote_date', 'interestRate','stock_price', 'strike', 'time_to_maturity', 'implied_volatility', 'callPrice']]
        
        
        #Create dataframes that are grouped by the quote date 
        self.dayGroups = self.params.groupby('quote_date')

        #Create dictionary where key is quote day (relative to earliest quote_date), 
        #and value is numpy-version of data frame associated with that quote date

        self.series_dict = {}
        for date, df in self.dayGroups:
            #First day is 9th
            first_day = 9
            self.series_dict[pd.to_datetime(date).day - first_day] = df.to_numpy()[:,1:].astype('float32')
 
        #Save preprocessed data
        all_series = self.gen_all_series()
        with open("data/processed_series.pkl", "wb") as f:
            pickle.dump(all_series, f)

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
        


            

    
    def __len__(self):
        return len(self.final_input)

    def __getitem__(self,i):


        '''
        Encoder-Transformer tries to predict implied volatilty, and the final model uses this with other parameters to predict call price.
        Therefore, the target is a tuple where the first item are the parameters to the Price MLP (strike, maturity, etc),
        and the second item is the associated call price the Price MLP is supposed to output
        '''

        input_data_padded = self.final_input[i]
        mask = self.final_masks[i]
        target_input = self.final_target_inputs[i]
        target_price = self.final_target_prices[i]

        return input_data_padded, mask, target_input, target_price


class finalDataset(Dataset):

    def __init__(self):
        self.df = pd.read_csv("data/UnderlyingOptionsEODQuotes_2024-10.csv")
        print("SELF>DF::", self.df)

        #self.df = pd.read_csv(self.data_csv)

        print(f"Initial rows: {len(self.df)}")
        #self.df = self.df.dropna()
        print(f"Rows after dropna: {len(self.df)}")



        # Calculate the target variable (mid price)
        bid_prices = self.df['bid_1545'].values
        ask_prices = self.df['ask_1545'].values
        self.target = (bid_prices + ask_prices) / 2  # Mid price as target
        print("TARGET PRICES:", self.target)

        # Calculate the stock price as the midpoint of underlying bid and ask prices
        self.df['stock_price'] = (self.df['underlying_bid_1545'] + self.df['underlying_ask_1545']) / 2
        print("STOCK PRICES:", self.df['stock_price'])


        # Calculate `time to maturity` in years from `expiration` and `quote_date`
        self.df['quote_date'] = pd.to_datetime(self.df['quote_date'])
        self.df['expiration'] = pd.to_datetime(self.df['expiration'])
        self.df['time_to_maturity'] = (self.df['expiration'] - self.df['quote_date']).dt.days / 365.0

        print("TIME TO MATURITY:", self.df['time_to_maturity'])


        implied_volatility = 0.2  # Default value for implied volatility
        self.df['implied_volatility'] = implied_volatility
        print("implied_volatility:", self.df['implied_volatility'])


        interestRate = 0.05  # Default value for implied volatility
        self.df['interestRate'] = implied_volatility
        print("interestRate:", self.df['interestRate'])


        #(interest rate, stock price, strike price, time, volatility), call price

        self.input = self.df[['interestRate','stock_price', 'strike', 'time_to_maturity', 'implied_volatility']].to_numpy()

        

        # Convert inputs and target to tensors
        self.input = torch.tensor(self.input, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32).reshape(-1, 1)
        
    
    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self,i):
       
        
        return self.input[i], self.target[i]

class SyntheticDataset(Dataset):

    def __init__(self):
        self.data_csv = "option_data.csv"
        print("OK:", self.data_csv)
        self.df = pd.read_csv(self.data_csv)
        self.data = self.df.to_numpy()[:, 1:]

        np.random.shuffle(self.data)

        self.data = torch.tensor(self.data,dtype=torch.float32)

        r = 0.05
        rates = torch.ones(len(self.data)) * r
        rates = rates.reshape(-1, 1)
        self.data = torch.cat((rates, self.data), dim=1)

        self.input = self.data[:, :5]
        self.target = self.data[:, 5].reshape(-1, 1)

        print("INPUT:", self.input)
        print("TARGET:", self.target)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        #(interest rate, stock price, strike price, time, volatility), call price
        return self.input[i], self.target[i]
    

class YahooDataset(Dataset):

    def __init__(self):
        self.data_csv = "data/calls_dataset.csv"
        self.df = pd.read_csv(self.data_csv)
        self.df = self.df.dropna()
        self.data = self.df.to_numpy()[:, 2:]
        self.data = self.data.astype(np.float32)

        np.random.shuffle(self.data)

        bid_prices = self.data[:, 1]
        ask_prices = self.data[:, 2]
        self.target = (bid_prices + ask_prices)/2


        self.target = torch.tensor(self.target,dtype=torch.float32).reshape(-1,1)

        self.data = np.delete(self.data, np.s_[1:3], axis=1)  
        self.data = torch.tensor(self.data,dtype=torch.float32)

        r = 0.0423
        rates = torch.ones(len(self.data)) * r
        rates = rates.reshape(-1, 1)
        self.data = torch.cat((rates, self.data), dim=1)

        self.input = self.data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        #(interest rate, strike price, volatility, time, stock price), price
        return self.input[i], self.target[i]
