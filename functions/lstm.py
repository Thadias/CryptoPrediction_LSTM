from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import time

ACTIV_FUNC = "linear"
LOSS = 'mae'
OPTIMIZER = 'adam'
DROPOUT = 0.25
NEURONS = 128

NAME = f'BTC_model-{int(time.time())}'
LOG_NAME = f'Model-logs{int(time.time())}'


def build_model(inputs, output_size):
    model = Sequential()

    model.add(LSTM(NEURONS, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=output_size, activation='softmax'))
    model.add(Activation(ACTIV_FUNC))

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['mae', 'acc'])

    print(f'Model finished compiling and it is saved as {NAME}')
    model.save(f'models/{NAME}')
    return model


def predict_model(model, data_inputs, data_output, pred_range,
                  test_input, data_test, window_size):
    tensorboard = TensorBoard(log_dir=f"logs/{LOG_NAME}")

    model_calc = model.fit(data_inputs[:-pred_range], data_output,
                           epochs=50,
                           batch_size=1,
                           verbose=2,
                           shuffle=True,
                           callbacks=[tensorboard])
    print(f'Model have been trained and saved as tensorboard logs - {LOG_NAME}')
    btc_pred = ((model.predict(data_test)[:-pred_range][::pred_range] + 1) *
                test_input['Close'].values[:-(window_size + pred_range)][::5].reshape(
                    int(np.ceil((len(data_test) - pred_range) / float(pred_range))), 1))
    print('BTC prediction prices calculated.')
    # Testing
    btc_pred = np.array(btc_pred)

    # print('Data in the Test_Input: ', test_input['Close'].values[:-(window_size + pred_range)][::5],
    #       '\n', 'Index: ', test_input.index[:-(window_size + pred_range)][::5])

    pred_array = []
    for pred in btc_pred:
        for price in pred:
            pred_array.append(price)

    date_list = pd.date_range('2019-02-10', '2019-03-01')
    pred_df = pd.DataFrame({'time': date_list, 'Pred': pred_array})
    pred_df.set_index('time', inplace=True)

    print('Predicetd price ready for a graphs')
    return pred_df

# def predict_prices(model, test_input, pred_range, data_test, window_size):
#
#     # pickle function need to add...
#     return btc_pred
