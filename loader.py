import pandas as pd
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self, train_data):        
        self.users = torch.tensor(train_data['user_id'].to_list())
        self.items = torch.tensor(train_data['product_id'].to_list())
        self.labels = torch.tensor(train_data['bought'].to_list())

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]