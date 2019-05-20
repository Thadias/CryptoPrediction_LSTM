import pandas as pd
import pickle
import os
import numpy as np
from functions import cryptoData


def preprocess_df(btc_df, split_time, window_size, norm_cols):
    # Sorting the main dataframe
    btc_df.sort_values(by='time')

    # Splitting the data to two datasets - the training set will have 90% of data, and test set the rest.
    training_set = btc_df[btc_df.index < split_time]
    test_set = btc_df[btc_df.index >= split_time]
    print('Training and test sets has been created')
    lstm_training_inputs = cryptoData.normalize_data(training_set, window_size, norm_cols)
    lstm_test_inputs = cryptoData.normalize_data(test_set, window_size, norm_cols)

    # pickle saving...
    # if

    # print('The data has been modified, normalized and temporary saved')
    print('Now the algorithm building begins...')
    return lstm_training_inputs, lstm_test_inputs, training_set, test_set


# Training output calculation
def data_output(data_training, data_test, window_size, pred_range):
    training_array = []
    for i in range(window_size, len(data_training['Close']) - pred_range):
        training_array.append((data_training['Close'][i:i + pred_range].values /
                                data_training['Close'].values[i - window_size]) - 1)

    test_output = (data_test['Close'][window_size:].values /
                   data_test['Close'][:-window_size].values) - 1
    training_output = np.array(training_array)

    print('The calculation of training and test Outputs are finished')
    return training_output, test_output
