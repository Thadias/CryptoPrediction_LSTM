from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard

import numpy as np

ACTIV_FUNC = 'linear'
LOSS = 'mae'
OPTIMIZER = 'adam'
DROPOUT = 0.25
NEURONS = 128


def build_model(inputs, output_size):
    model = Sequential()

    model.add(LSTM(NEURONS, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=output_size))
    model.add(ACTIV_FUNC)

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['mae', 'acc'])
    return model


def fit_model(model, data_inputs, data_output, pred_range, tensorboard):
    model_calc = model.fit(data_inputs[:-pred_range], data_output,
                           epochs=50,
                           batch_size=1,
                           verbose=2,
                           shuffle=True,
                           callbacks=[tensorboard])

    return model_calc


def predict_prices(model, test_input, pred_range, data_test, window_size):
    btc_pred = ((model.predict(data_test)[:-pred_range][::pred_range] + 1) *
                test_input['Close'].values[:-(window_size + pred_range)][::5].reshape(
                    int(np.ceil((len(test_input) - pred_range) / float(pred_range))), 1))
    # pickle function need to add...
    return btc_pred
