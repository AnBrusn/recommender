import pandas as pd
import numpy as np
import pickle
from random import randrange       

def apk(actual, predicted, k=10):
    '''Calculate average precision at k.'''
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    '''Calculate mean average precision at k.'''
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def hitk(actual, predicted, k=10):
    '''Calculate hits at k.'''
    hits = []
    for a, p in zip(actual, predicted):
        if len(p) > k:
            p = p[:k]
        hits += [len(np.intersect1d(a, p)) / min(len(p), k)]
    return np.mean(hits)


def _split_indices(grouped_ratings, retriever):
    return np.concatenate(grouped_ratings.apply(retriever).values)


def split(orders):
    '''
    Get order indices for train, validatio and test.
    Train - 50% of orders, validation - 25%, test - 25%
    '''
    grouper = orders.sort_values('order_number').groupby('user_id')
    train_indices = _split_indices(
        grouper,
        lambda user_ratings: user_ratings[:int(user_ratings.shape[0] * 0.5)]['order_id'].values)
    
    validation_indices = _split_indices(
        grouper,
        lambda user_ratings: user_ratings.iloc[int(user_ratings.shape[0] * 0.5):
                                               int(user_ratings.shape[0] * 0.75)]['order_id'].values)
    
    test_indices = _split_indices(
        grouper,
        lambda user_ratings: user_ratings.iloc[int(user_ratings.shape[0] * 0.75):]['order_id'].values)
    
    with open('saves/train_ids.pickle', 'wb') as output:
        pickle.dump(train_indices, output)
    with open('saves/test_ids.pickle', 'wb') as output:
        pickle.dump(test_indices, output)
    with open('saves/validation_ids.pickle', 'wb') as output:
        pickle.dump(validation_indices, output)
    
    return train_indices, validation_indices, test_indices

    
def load_data():    
    '''Create user-product dataframes for train, validation and test.'''
    with open('saves/train_ids.pickle', 'rb') as input:
        train_ids = pickle.load(input)
    with open('saves/test_ids.pickle', 'rb') as input:
        test_ids = pickle.load(input)
    with open('saves/validation_ids.pickle', 'rb') as input:
        validation_ids = pickle.load(input)
        
    orders = pd.read_csv('orders.csv')[['user_id', 'order_id']]
    
    order_products_prior = pd.read_csv('order_products__prior.csv')[['product_id', 'order_id']]
    order_products_train = pd.read_csv('order_products__train.csv')[['product_id', 'order_id']]
    order_products = pd.concat([order_products_prior, order_products_train])
    orders = pd.merge(order_products, orders, on = 'order_id')
    orders = orders[orders['order_id'].isin(np.concatenate([train_ids, test_ids, validation_ids]))]
    
    user_enum = {i: x for x, i in enumerate(orders['user_id'].unique())}
    orders['user_id'] = orders['user_id'].map(user_enum)
    product_enum = {i: x for x, i in enumerate(orders['product_id'].unique())}
    orders['product_id'] = orders['product_id'].map(product_enum)
    
    train_data = orders[orders['order_id'].isin(train_ids)].drop('order_id', 1)
    train_data = train_data.groupby(train_data.columns.tolist()).size().reset_index().rename(columns={0:'bought'})
    validation_data = orders[orders['order_id'].isin(validation_ids)].drop('order_id', 1)
    validation_data = validation_data.groupby(validation_data.columns.tolist()).size().reset_index().reset_index().rename(columns={0:'bought'})
    test_data = orders[orders['order_id'].isin(test_ids)].drop('order_id', 1)
    test_data = test_data.groupby(test_data.columns.tolist()).size().reset_index().rename(columns={0:'bought'})
    
    return train_data, validation_data, test_data, user_enum, product_enum
