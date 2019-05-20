from functions import cryptoData, lstm, preprocess
import time

DIR_BACKUP = './backup/'
DIR_DATA = './data/'
DIR_FUNCTIONS = './functions'

SPLIT_TIME = '2019-01-28'
WINDOW_SIZE = 10
NORM_COLS = ['Close', 'Volume']
PRED_RANGE = 5

start_pgm = time.time()
print('The program begins!! ')
main_df = cryptoData.load_data()
# clean_df = cryptoData.modify_data(main_df)

lstm_training_input, lstm_test_input, training_set, test_set = \
    preprocess.preprocess_df(main_df, SPLIT_TIME, WINDOW_SIZE, NORM_COLS)
lstm_training_output, lstm_test_output = preprocess.data_output(training_set, test_set,
                                                                WINDOW_SIZE, PRED_RANGE)
print('\n')
btc_model = lstm.build_model(lstm_training_input, output_size=PRED_RANGE)
btc_pred_prices = lstm.predict_model(btc_model, lstm_training_input, lstm_training_output,
                                     PRED_RANGE, test_set, lstm_test_input, WINDOW_SIZE)

end_pgm = start_pgm - time.time()
print(f'Time for complete the program - {int(end_pgm)} seconds',
      '\n', 'Program has ended.')
