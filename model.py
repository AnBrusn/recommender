import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from loader import TrainDataset

class NCF(pl.LightningModule):
    """
    Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
    """
    
    def __init__(self, dim, num_users, num_items, train_data):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(num_items, dim)
        self.fc1 = nn.Linear(in_features=dim*2, out_features=dim*5)
        self.fc2 = nn.Linear(in_features=dim*5, out_features=dim*3)
        self.fc3 = nn.Linear(in_features=dim*3, out_features=dim)
        self.output = nn.Linear(in_features=dim, out_features=1)
        self.train_data = train_data
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        vector = nn.ReLU()(self.fc3(vector))
        pred = nn.Sigmoid()(self.output(vector))
        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(TrainDataset(self.train_data), batch_size=1024, num_workers=4)