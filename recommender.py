import pandas as pd
import numpy as np
import pickle
from random import randrange
from model import NCF
import torch
import pytorch_lightning as pl

class Recommender():
    
    def __init__(self, train_data=None, validation_data=None, test_data=None, neg_num=1):
        self.components_number = 100
        if train_data is not None:
            self.num_users = train_data['user_id'].unique().size;
            self.num_items = np.unique(np.concatenate((train_data['product_id'].to_numpy(),
                                                        validation_data['product_id'].to_numpy(), 
                                                        test_data['product_id'].to_numpy()))).size
            self.get_data_for_model(train_data, validation_data, test_data, neg_num)
                
                
    def set_components_number(self, components_number):
        '''
        Set components number for embedding
        '''
        self.components_number = components_number
                
            
    def set_model(self, path_to_svd):
        '''
        Load model from file.
        
        path_to_svd -- path to file with svd-model
        '''
        with open(path_to_svd, 'rb') as inp:
            self.model = pickle.load(inp)
            
            
    def get_data_for_model(self, train_data, validation_data, test_data, neg_num):
        '''Prepare data, collect negative items'''
        user_interacted_items = train_data.groupby('user_id')['product_id'].apply(list).to_dict()
        users = train_data['user_id'].to_list()
        items = train_data['product_id'].to_list()
        bought = [1]  * len(users)    
        for (user, interacted_items) in user_interacted_items.items():
            not_interacted_items = set(range(self.num_items)) - set(interacted_items)
            negative_items = list(np.random.choice(list(not_interacted_items), neg_num*len(interacted_items)))
            users += [user] * neg_num*len(interacted_items)
            items += negative_items
            bought += [0] * neg_num*len(interacted_items)
        self.train_data = pd.DataFrame(
            np.column_stack((users, items, bought)),
            columns = ['user_id', 'product_id', 'bought']
        )
        
        user_interacted_items = validation_data.groupby('user_id')['product_id'].apply(list).to_dict()
        users = validation_data['user_id'].to_list()
        items = validation_data['product_id'].to_list()
        bought = [1]  * len(users)      
        for (user, interacted_items) in user_interacted_items.items():
            not_interacted_items = set(range(self.num_items)) - set(interacted_items)
            negative_items = list(np.random.choice(list(not_interacted_items), len(interacted_items) * 10))
            users += [user] * len(interacted_items) * 10
            items += negative_items
            bought += [0] * len(interacted_items) * 10
        self.val_data = pd.DataFrame(
            np.column_stack((users, items, bought)), 
            columns = ['user_id', 'product_id', 'bought']
        )
        
        user_interacted_items = test_data.groupby('user_id')['product_id'].apply(list).to_dict()
        users = test_data['user_id'].to_list()
        items = test_data['product_id'].to_list()
        bought = [1]  * len(users)         
        for (user, interacted_items) in user_interacted_items.items():
            not_interacted_items = set(range(self.num_items)) - set(interacted_items)
            negative_items = list(np.random.choice(list(not_interacted_items), len(interacted_items) * 10))
            users += [user] * len(interacted_items) * 10
            items += negative_items
            bought += [0] * len(interacted_items) * 10
        self.test_data = pd.DataFrame(
            np.column_stack((users, items, bought)), 
            columns = ['user_id', 'product_id', 'bought']
        )
        
        
    def set_data_for_model(self, path_to_train, path_to_val, path_to_test):
        '''Load prepared data from files.'''
        self.train_data = pd.read_csv(path_to_train)
        self.val_data = pd.read_csv(path_to_val)
        self.test_data = pd.read_csv(path_to_test)
        self.num_users = self.train_data['user_id'].unique().size;
        self.num_items = np.unique(np.concatenate((self.train_data['product_id'].to_numpy(),
                                                    self.val_data['product_id'].to_numpy(), 
                                                    self.test_data['product_id'].to_numpy()))).size

        
    def fit(self, path_to_save, epochs=4):
        '''
        Train ncf-model.
        
        path_to_save -- path to file to save model (if False do not save)
        '''
        self.model = NCF(self.components_number, self.num_users, self.num_items, self.train_data)
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model)
        
        if path_to_save:
            with open(path_to_save, 'wb') as out:
                pickle.dump(self.model, out)      
    
    
    def predict(self, user_ids, item_ids):
        '''Get predictions for users and items.'''
        pred_tensor = self.model(torch.tensor(user_ids), torch.tensor(item_ids))
        return pred_tensor.detach().numpy().reshape(1, pred_tensor.shape[0])[0]
    
    
    def predict_for_user(self, user_id, item_ids):
        '''Get predictions for one user and all items from list.'''
        user_ids = [user_id] * len(item_ids)
        pred_tensor = self.model(torch.tensor(user_ids), torch.tensor(item_ids))
        return pred_tensor.detach().numpy().reshape(1, pred_tensor.shape[0])[0]
        