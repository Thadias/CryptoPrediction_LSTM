from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def load_data(filename, sequence_length):
    """
    Loads the bitcoin data

    Arguments:
    filename -- A string that represents where the .csv file can be located
    sequence_length -- An integer of how many days should be looked at in a row

    Returns:
    X_train -- A tensor of shape (2400, 49, 35) that will be inputed into the model to train it
    Y_train -- A tensor of shape (2400,) that will be inputed into the model to train it
    X_test -- A tensor of shape (267, 49, 35) that will be used to test the model's proficiency
    Y_test -- A tensor of shape (267,) that will be used to check the model's predictions
    Y_daybefore -- A tensor of shape (267,) that represents the price of bitcoin the day before each Y_test value
    unnormalized_bases -- A tensor of shape (267,) that will be used to get the true prices from the normalized ones
    window_size -- An integer that represents how many days of X values the model can look at at once
    """
    # Read the data file
    raw_data = pd.read_csv(filename, dtype=float).values

    # Change all zeros to the number before the zero occurs
    for x in range(0, raw_data.shape[0]):
        for y in range(0, raw_data.shape[1]):
            if (raw_data[x][y] == 0):
                raw_data[x][y] = raw_data[x - 1][y]

    # Convert the file to a list
    data = raw_data.tolist()

    # Convert the data to a 3D array (a x b x c)
    # Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    # Normalizing data by going through each window
    # Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    dr[:, 1:, :] = d0[:, 1:, :] / d0[:, 0:1, :] - 1

    # Keeping the unnormalized prices for Y_test
    # Useful when graphing bitcoin price over time later
    start = 2400
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end, 0:1, 20]

    # Splitting data set into training (First 90% of data points) and testing data (last 10% of data points)
    split_line = round(0.9 * dr.shape[0])
    training_data = dr[:int(split_line), :]

    # Shuffle the data
    np.random.shuffle(training_data)

    # Training Data
    X_train = training_data[:, :-1]
    Y_train = training_data[:, -1]
    Y_train = Y_train[:, 20]

    # Testing data
    X_test = dr[int(split_line):, :-1]
    Y_test = dr[int(split_line):, 49, :]
    Y_test = Y_test[:, 20]

    # Get the day before Y_test's price
    Y_daybefore = dr[int(split_line):, 48, :]
    Y_daybefore = Y_daybefore[:, 20]

    # Get window size and sequence length
    sequence_length = sequence_length
    window_size = sequence_length - 1  # because the last value is reserved as the y value

    return X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size


def initialize_model(window_size, dropout_value, activation_function, loss_function, optimizer):
    """
    Initializes and creates the model to be used

    Arguments:
    window_size -- An integer that represents how many days of X_values the model can look at at once
    dropout_value -- A decimal representing how much dropout should be incorporated at each level, in this case 0.2
    activation_function -- A string to define the activation_function, in this case it is linear
    loss_function -- A string to define the loss function to be used, in the case it is mean squared error
    optimizer -- A string to define the optimizer to be used, in the case it is adam

    Returns:
    model -- A 3 layer RNN with 100*dropout_value dropout in each layer that uses activation_function as its activation
             function, loss_function as its loss function, and optimizer as its optimizer
    """
    # Create a Sequential model using Keras
    model = Sequential()
    # First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train.shape[-1]), ))
    model.add(Dropout(dropout_value))
    # Second recurrent layer with dropout
    model.add(Bidirectional(LSTM((window_size * 2), return_sequences=True)))
    model.add(Dropout(dropout_value))
    # Third recurrent layer
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    # Output layer (returns the predicted value)
    model.add(Dense(units=1))
    # Set activation function
    model.add(Activation(activation_function))
    # Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae', 'acc'])

    return model


def fit_model(model, X_train, Y_train, batch_num, num_epoch, val_split):
    LOG_NAME = f'CompA1-logs{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{LOG_NAME}')
    # Record the time the model starts training
    start = time.time()
    # Train the model on X_train and Y_train
    model.fit(X_train, Y_train,
              batch_size=batch_num,
              nb_epoch=num_epoch,
              validation_split=val_split,
              callbacks=[tensorboard]
              )
    # Get the time it took to train the model (in seconds)
    training_time = int(math.floor(time.time() - start))
    return model, training_time


def test_model(model, X_test, Y_test, unnormalized_bases):
    # Test the model on X_Test
    y_predict = model.predict(X_test)
    # Create empty 2D arrays to store unnormalized values
    real_y_test = np.zeros_like(Y_test)
    real_y_predict = np.zeros_like(y_predict)
    # Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
    for i in range(Y_test.shape[0]):
        y = Y_test[i]
        predict = y_predict[i]
        real_y_test[i] = (y + 1) * unnormalized_bases[i]
        real_y_predict[i] = (predict + 1) * unnormalized_bases[i]

    # Plot of the predicted prices versus the real prices
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title("Bitcoin Price Over Time")
    plt.plot(real_y_predict, color='green', label='Predicted Price')
    plt.plot(real_y_test, color='red', label='Real Price')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (Days)")
    ax.legend()

    return y_predict, real_y_test, real_y_predict, fig


def price_change(Y_daybefore, Y_test, y_predict):
    # Reshaping Y_daybefore and Y_test
    Y_daybefore = np.reshape(Y_daybefore, (-1, 1))
    Y_test = np.reshape(Y_test, (-1, 1))
    # The difference between each predicted value and the value from the day before
    delta_predict = (y_predict - Y_daybefore) / (1 + Y_daybefore)
    # The difference between each true value and the value from the day before
    delta_real = (Y_test - Y_daybefore) / (1 + Y_daybefore)
    # Plotting the predicted percent change versus the real percent change
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Percent Change in Bitcoin Price Per Day")
    plt.plot(delta_predict, color='green', label='Predicted Percent Change')
    plt.plot(delta_real, color='red', label='Real Percent Change')
    plt.ylabel("Percent Change")
    plt.xlabel("Time (Days)")
    ax.legend()
    plt.show()

    return Y_daybefore, Y_test, delta_predict, delta_real, fig


def binary_price(delta_predict, delta_real):
    delta_predict_1_0 = np.empty(delta_predict.shape)
    delta_real_1_0 = np.empty(delta_real.shape)

    # If the change in price is greater than zero, store it as a 1
    # If the change in price is less than zero, store it as a 0
    for i in range(delta_predict.shape[0]):
        if delta_predict[i][0] > 0:
            delta_predict_1_0[i][0] = 1
        else:
            delta_predict_1_0[i][0] = 0
    for i in range(delta_real.shape[0]):
        if delta_real[i][0] > 0:
            delta_real_1_0[i][0] = 1
        else:
            delta_real_1_0[i][0] = 0

    return delta_predict_1_0, delta_real_1_0


def find_positives_negatives(delta_predict_1_0, delta_real_1_0):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(delta_real_1_0.shape[0]):
        real = delta_real_1_0[i][0]
        predicted = delta_predict_1_0[i][0]
        if real == 1:
            if predicted == 1:
                true_pos += 1
            else:
                false_neg += 1
        elif real == 0:
            if predicted == 0:
                true_neg += 1
            else:
                false_pos += 1
    return true_pos, false_pos, true_neg, false_neg


def calculate_statistics(true_pos, false_pos, true_neg, false_neg, y_predict, Y_test):
    precision = float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)
    F1 = float(2 * precision * recall) / (precision + recall)
    # Get Mean Squared Error
    MSE = mean_squared_error(y_predict.flatten(), Y_test.flatten())

    return precision, recall, F1, MSE


X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size = load_data("Bitcoin Data.csv", 50)

model = initialize_model(window_size, 0.2, 'linear', 'mse', 'adam')
model, training_time = fit_model(model, X_train, Y_train, 1024, 50, .05)


y_predict, real_y_test, real_y_predict, fig1 = test_model(model, X_test, Y_test, unnormalized_bases)
#Show the plot
plt.show(fig1)

Y_daybefore, Y_test, delta_predict, delta_real, fig2 = price_change(Y_daybefore, Y_test, y_predict)
#Show the plot
plt.show(fig2)