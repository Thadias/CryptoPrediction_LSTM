import pandas as pd
import numpy as np

FILE_NAME = 'BITSTAMPEUR.csv'


def load_data():
    orig_btc_df = pd.read_csv(f"./data/{FILE_NAME}")
    # Clearing the columns names
    orig_btc_df.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_CUR', 'Weighted_Price']
    # Set index for the Dataframe. Index is Time.
    orig_btc_df.set_index("time", inplace=True)
    # Executing functions, which modifies the data for the preprocess of algorithm.
    btc_df = modify_data(orig_btc_df)

    print('The data has been loaded into program variables')
    return btc_df


def modify_data(original_df):
    # new_df = original_df
    # Adding two news columns: 'Close_Off_High' - daily average;
    # 'Votalitily' - The difference when compared to previous date.
    kwargs = {'close_off_high': lambda x: 2 * (x['High'] - x['Close']) / (x['High'] - x['Low']) - 1,
              'volatility': lambda x: (x['High'] - x['Low'] / x['Open'])}
    # Assign the two new columns to the main dataset
    new_df = original_df.assign(**kwargs)
    # Removing the rows with values - 'NaN' or 'inf'.
    new_df.dropna(inplace=True)
    # Leaving only the important columns
    new_df = new_df[['Close', 'Volume', 'close_off_high', 'volatility']]

    print('The data has been modified and prepare for program to process it')
    return new_df


def normalize_data(data_set, window_size, norm_col):
    data_inputs = []
    # For each row in the data set.
    # The normalizing must be done. Which means that the value is between -1 and 1, when comparing
    # to first value of window
    for i in range(len(data_set) - window_size):
        temp_set = data_set[i:(i + window_size)].copy()
        for col in norm_col:
            temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
        data_inputs.append(temp_set)

    data_inputs = [np.array(data_input) for data_input in data_inputs]
    data_inputs = np.array(data_inputs)

    print('Returning data inputs')
    return data_inputs





# Function to save the pickle dump and check if it's there.
# def pickle_data(file_name, data):
#     # data_inputs = []
#     if os.path.isfile(f'./pickles/{file_name}.pkl'):
#         with open(f'{file_name}.pkl', 'rb') as f:
#             data_inputs = pickle.load(f)
#         print(f'Data has been get from file {file_name}.pkl')
#         return data_inputs
#     else:
#         with open(f'{file_name}.pkl', 'wb') as f:
#             pickle.dump(data, f)
#         print(f'Data has been dumped into {file_name}.pkl file')
