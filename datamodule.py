import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import warnings
from tqdm import tqdm
import random
warnings.filterwarnings(action='ignore')

class FATData(Dataset) :
    def __init__(self, x, y) :
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        
    def __len__(self) :
        return self.x.__len__()

    def __getitem__(self, idx) :
        return self.x[idx], self.y[idx]

    


class FATDataModule(pl.LightningDataModule) :
    def __init__ (self, clf_stock=[], seq_len = 30 , batch_size = 128, num_workers = 0) :
        super().__init__()
        self.clf_stock = clf_stock
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.columns = None
        self.scaler = None
        # TODO: change up, down percent list and MA list
        self.up_per_list = [1, 3, 5]
        self.down_per_list = [-1, -3, -5]
        self.MA_list = [5, 20]
        self.window_size = 30
        self.split_ratio = [0.8, 0.1, 0.1]
        
    def setup(self, stage=None) :     
        # Load and Preprocess data
        print('Preprocessing Data...')
        # define columns
        x_columns = ['TIME', 'CLOSE', 'HIGH', 'LOW', 'VOL']
        for MA in self.MA_list:
            x_columns.append('MA{}'.format(MA))

        y_columns = []
        for up_per in self.up_per_list:
            y_columns.append('UP_{}PER'.format(up_per))
        for down_per in self.down_per_list:
            y_columns.append('DOWN_{}PER'.format(down_per))

        scale_columns = ['CLOSE', 'HIGH', 'LOW', 'VOL']
        for MA in self.MA_list:
            scale_columns.append('MA{}'.format(MA))
        
        train_x_np_list = []
        train_y_np_list = []
        val_x_np_list = []
        val_y_np_list = []
        test_x_np_list = []
        test_y_np_list = []
        
        for stock in tqdm(self.clf_stock):
            stock_df = pd.read_csv('./DATA/min/{}_concat.csv'.format(stock))
            stock_df['FUTURE_HIGH'] = stock_df['HIGH']
            stock_df['FUTURE_LOW'] = stock_df['LOW']
            for index in stock_df.index:
                if stock_df['TIME'][index] != 1530 :
                    stock_df['FUTURE_HIGH'][index] = max(stock_df['FUTURE_HIGH'][index-1], stock_df['FUTURE_HIGH'][index])
                    stock_df['FUTURE_LOW'][index] = min(stock_df['FUTURE_LOW'][index-1], stock_df['FUTURE_LOW'][index])
            stock_df = stock_df[::-1].reset_index(drop=True)
            
            # Make UP, DOWN Label Data
            for up_per in self.up_per_list:
                stock_df['UP_{}PER'.format(up_per)] = (stock_df['FUTURE_HIGH'] > (1+0.01*up_per)*stock_df['CLOSE']).astype(int)
            for down_per in self.down_per_list:
                stock_df['DOWN_{}PER'.format(down_per)] = (stock_df['FUTURE_LOW'] < (1+0.01*down_per)*stock_df['CLOSE']).astype(int)
            
            # Make MA Data
            daily_stock_df = pd.read_csv('.\DATA\\day(3month)\\{}DAY_3month.csv'.format(stock), parse_dates=['DATE'])
            daily_stock_df = daily_stock_df[::-1].reset_index(drop=True)
            for MA in self.MA_list:
                daily_stock_df['MA{}'.format(MA)] = daily_stock_df['CLOSE'].rolling(window=MA).mean()
                
            # merge daily data
            stock_df['DATE'] = pd.to_datetime(stock_df['DATE'], format='%Y%m%d')
            daily_stock_df['DATE'] = pd.to_datetime(daily_stock_df['DATE'])
            stock_df = pd.merge(stock_df, daily_stock_df, on='DATE', how='left', suffixes=('','_daily'))
            stock_df = stock_df[x_columns+y_columns]
            
            scaler = MinMaxScaler()
            scaler.fit(stock_df[scale_columns])
            stock_df[scale_columns] = pd.DataFrame(scaler.transform(stock_df[scale_columns]))
            
            stock_x_df = stock_df[x_columns]
            stock_y_df = stock_df[y_columns]
            print(stock_x_df.describe())
            print(stock_y_df.describe())
            
            stock_x_np = stock_x_df.values
            stock_y_np = stock_y_df.values
            
            # make sequence data
            x_list = []
            y_list = []
            for i in range(len(stock_df) - self.window_size):
                x_list.append(stock_x_np[i:i+self.window_size])
                y_list.append(stock_y_np[i+self.window_size])
            x_np = np.array(x_list)
            y_np = np.array(y_list)
            
            index = [i for i in range(len(x_np))]    
            x_train = x_np[index[:round(len(index)*self.split_ratio[0])]]
            y_train = y_np[index[:round(len(index)*self.split_ratio[0])]]
            x_val = x_np[index[round(len(index)*self.split_ratio[0]):round(len(index)*(self.split_ratio[0]+self.split_ratio[1]))]]
            y_val = y_np[index[round(len(index)*self.split_ratio[0]):round(len(index)*(self.split_ratio[0]+self.split_ratio[1]))]]
            x_test = x_np[index[round(len(index)*(self.split_ratio[0]+self.split_ratio[1])):]]
            y_test = y_np[index[round(len(index)*(self.split_ratio[0]+self.split_ratio[1])):]]
            
            train_x_np_list.append(x_train)
            train_y_np_list.append(y_train)
            val_x_np_list.append(x_val)
            val_y_np_list.append(y_val)
            test_x_np_list.append(x_test)
            test_y_np_list.append(y_test)
        
        self.x_train = np.concatenate(train_x_np_list, axis=0)
        self.y_train = np.concatenate(train_y_np_list, axis=0)
        self.x_val = np.concatenate(val_x_np_list, axis=0)
        self.y_val = np.concatenate(val_y_np_list, axis=0)
        self.x_test = np.concatenate(test_x_np_list, axis=0)
        self.y_test = np.concatenate(test_y_np_list, axis=0)
        
    def train_dataloader(self):
        print('Training a model...')
        train_dataset = FATData(self.x_train, 
                                self.y_train)
        train_loader = DataLoader(train_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = FATData(self.x_val, 
                                self.y_val)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)
        return val_loader  

    def test_dataloader(self):
        print('Testing the model...')
        test_dataset = FATData(self.x_test, 
                                self.y_test)
        test_loader = DataLoader(test_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)
        return test_loader
