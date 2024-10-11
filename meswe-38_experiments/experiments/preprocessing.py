import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
import torch
from typing import Tuple


def create_sequences(data:np.ndarray, time_steps:int, sliding_window:int) -> Tuple[np.ndarray, np.ndarray]:
    sequences, targets = [], []
    for i in range(len(data) - time_steps - sliding_window + 1):
        sequences.append(data[i:i + time_steps, :])
        targets.append(data[i + time_steps + sliding_window-1, -1]) 
    return np.array(sequences), np.array(targets)


def load_data(file_ids: list[str], time_steps: int=100, sliding_window:int=1, folder_path:str='MESWE-38/MESWE-38-processed/') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV files into separate DataFrames for training, validation and test.
    """

    numpy_data_x = []
    numpy_data_y = []

    for id in file_ids:
        file_path = f"{folder_path}meswe_event_{id}.csv"
        df = pd.read_csv(file_path)
        cols = df.columns
        
        if "Timestamp" in cols:
            df = df.drop(["Timestamp"], axis=1) 

        # move dst to last col
        df['Dst'] = df.pop('Dst')
        
        df = df.dropna()
        
        np_data = df.to_numpy()
        data_X, data_y = create_sequences(np_data, time_steps, sliding_window)
        numpy_data_x.append(data_X)
        numpy_data_y.append(data_y)
    
    
    data_X_concated = np.concatenate(numpy_data_x, axis=0)
    data_y_concated = np.concatenate(numpy_data_y, axis=0)

    return data_X_concated, data_y_concated



def create_graph_from_sequence(sequence: torch.Tensor, target: torch.Tensor) -> Data:
    """
    Create a graph with edge index from a single sequence
    sequence shape: (seq_len, num_features)
    target shape: (1, 1)
    """
    num_nodes = sequence.shape[0]
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long).t().contiguous()
    return Data(x=sequence, edge_index=edge_index, y=target)


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return create_graph_from_sequence(self.X[idx], self.y[idx])
    

def get_k_fold(k_fold: int) -> Tuple[list[int], list[int], list[int]]:
    """function returns event number for train, validation and test set. 
    This numbers are then processed and DataFrame is created out of them.
    """

    if k_fold == 1:
        file_numbers_train = [1, 4, 6]
        file_numbers_val = [7, 8, 9]  
        file_numbers_test = [11, 12, 13] 


    elif k_fold == 2:
        file_numbers_train = [21, 23, 25]
        file_numbers_val = [30, 31, 32]  
        file_numbers_test = [11, 29, 30] 

    
    elif k_fold == 3:
        file_numbers_train = [29, 30, 9]
        file_numbers_val = [4, 11, 32]  
        file_numbers_test = [37, 12, 13] 


    elif k_fold == 4:
        file_numbers_train = [11, 14, 15]
        file_numbers_val = [12, 13, 1]  
        file_numbers_test = [35, 36, 37] 
    
    else:
        raise Exception("wrong k-fold selected")
    
    return file_numbers_train, file_numbers_val, file_numbers_test


def get_torch_data(data_X: np.ndarray, data_y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    data_X: torch.Tensor = torch.from_numpy(data_X)
    data_y: torch.Tensor = torch.from_numpy(data_y)
    # this is to get same dimensions for X and y data
    data_y = data_y.unsqueeze(1).unsqueeze(1)
    
    data_X = data_X.float()
    data_y = data_y.float()
    return data_X, data_y




class StandardScaler():
    def __init__(self, train_X, train_y):

        self.data_mean = np.mean(train_X, axis=(0, 1), keepdims=True)
        self.data_std = np.std(train_X, axis=(0, 1), keepdims=True)

        self.y_mean = np.mean(train_y, keepdims=True)
        self.y_std = np.std(train_y, keepdims=True)

    def standardize_X(self, data_X: np.ndarray) -> np.ndarray:
        """ standardizes input data with mean and std from train set"""
        data_X_scaled = (data_X - self.data_mean) / self.data_std
        return data_X_scaled
        
    def standardize_y(self, data_y: np.ndarray) -> np.ndarray:
        """ standardizes target data with mean and std from train set"""
        data_y_scaled = (data_y - self.y_mean) / self.y_std
        return data_y_scaled



