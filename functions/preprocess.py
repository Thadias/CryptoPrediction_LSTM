import pandas as pd
import pickle
import os
from functions import cryptoData

DIR_BACKUP = './backup/'
DIR_DATA = './data/'
DIR_FUNCTIONS = './functions'

SPLIT_TIME = '2019-01-28'
WINDOW_SIZE = 10
NORM_COLS = ['Close', 'Volume']

PRED_RANGE = 5


def preprocess_df(btc_df):
    # Sorting the main dataframe
    btc_df.sort_values(by='time')

    # Splitting the data to two datasets - the training set will have 90% of data, and test set the rest.
    training_set = btc_df[btc_df.index < SPLIT_TIME]
    test_set = btc_df[btc_df.index >= SPLIT_TIME]
    print('Training and test sets has been created')
    lstm_training_inputs = cryptoData.normalize_data(training_set, WINDOW_SIZE, NORM_COLS)
    lstm_test_inputs = cryptoData.normalize_data(test_set, WINDOW_SIZE, NORM_COLS)

    # pickle saving...
    # if

    print('The data has been modified, normalized and temporary saved')
    print('Now the algorithm building begins...')





