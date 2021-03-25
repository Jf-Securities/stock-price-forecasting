'''
Script to run time series models against all data in the \data folder.

Author: @josh
'''
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from fbprophet import Prophet
from keras.layers import Dense, Dropout, GRU, SimpleRNN
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

plt.style.use('fivethirtyeight')

DAYS = 10


def create_files_dict(pth='./data/'):
    '''
    create dictionary of files
    '''
    # pull all data files
    files = os.listdir(pth)
    print(files)

    all_data = dict()
    for file in files:
        # create key and file path
        file_key = file.split('_')[0]
        file_path = os.path.join(pth, file)

        # read the data
        data = pd.read_csv(
            file_path,
            index_col='Date',
            parse_dates=['Date']
        )

        # store data in dictionary
        all_data[file_key] = data

    return all_data


def create_dl_train_test_split(all_data):
    '''
    create training/testing data and scaler object
    '''
    # create training and test set
    training_set = all_data[:-DAYS].iloc[:, 1:2].values
    test_set = all_data[-DAYS:(-DAYS + 1)].iloc[:, 1:2].values

    # scale the data
    sc = MinMaxScaler(feature_range=(0, 1))
    if len(training_set) < 200:
        return None, None, None, None
    training_set_scaled = sc.fit_transform(training_set)

    # create training and test data
    X_train = []
    y_train = []
    for i in range(60, len(training_set)):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    total_data = all_data["High"]
    inputs = total_data[len(total_data) - len(test_set) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(60, 61):
        X_test.append(inputs[i - 60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc


def create_single_layer_small_rnn_model(X_train, y_train):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    model = None
    if model is None:
        model = Sequential()
        model.add(SimpleRNN(6))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    model.fit(X_train, y_train, epochs=100, batch_size=150)

    return model


def predict_from_model(model, X_test):
    scaled_preds = model.predict(X_test)
    return scaled_preds


def create_single_layer_rnn_model(X_train, y_train):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    model = None
    if model is None:
        model = Sequential()
        model.add(SimpleRNN(32))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    model.fit(X_train, y_train, epochs=100, batch_size=150)

    return model


def create_rnn_model(X_train, y_train):
    '''
    create rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    model = None
    if model is None:
        model = Sequential()
        model.add(SimpleRNN(32, return_sequences=True))
        model.add(SimpleRNN(32, return_sequences=True))
        model.add(SimpleRNN(32, return_sequences=True))
        model.add(SimpleRNN(32))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    model.fit(X_train, y_train, epochs=100, batch_size=150)

    return model


def create_GRU_model(X_train, y_train):
    '''
    create GRU model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # The GRU architecture
    regressorGRU = None
    if regressorGRU is None:
        regressorGRU = Sequential()
        # First GRU layer with Dropout regularisation
        regressorGRU.add(GRU(units=50, return_sequences=True,
                             input_shape=(X_train.shape[1], 1), activation='tanh'))
        regressorGRU.add(GRU(units=50, return_sequences=True, activation='tanh'))
        regressorGRU.add(GRU(units=50, return_sequences=True, activation='tanh'))
        regressorGRU.add(GRU(units=50, activation='tanh'))
        regressorGRU.add(Dense(units=1))

        # Compiling the RNN
        regressorGRU.compile(
            optimizer=SGD(
                lr=0.01,
                decay=1e-7,
                momentum=0.9,
                nesterov=False),
            loss='mean_squared_error')
    # Fitting to the training set
    regressorGRU.fit(X_train, y_train, epochs=50, batch_size=150)

    return regressorGRU


def predict_gru(model, X_test):
    GRU_predicted_stock_price = model.predict(X_test)
    return GRU_predicted_stock_price


def create_GRU_with_drop_out_model(X_train, y_train):
    '''
    create GRU model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # The GRU architecture
    regressorGRU = None
    if regressorGRU is None:
        regressorGRU = Sequential()
        # First GRU layer with Dropout regularisation
        regressorGRU.add(GRU(units=50, return_sequences=True,
                             input_shape=(X_train.shape[1], 1), activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # Second GRU layer
        regressorGRU.add(GRU(units=50, return_sequences=True, activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # Third GRU layer
        regressorGRU.add(GRU(units=50, return_sequences=True, activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # Fourth GRU layer
        regressorGRU.add(GRU(units=50, activation='tanh'))
        regressorGRU.add(Dropout(0.2))
        # The output layer
        regressorGRU.add(Dense(units=1))
        # Compiling the RNN
        regressorGRU.compile(
            optimizer=SGD(
                lr=0.01,
                decay=1e-7,
                momentum=0.9,
                nesterov=False),
            loss='mean_squared_error')
    # Fitting to the training set
    regressorGRU.fit(X_train, y_train, epochs=50, batch_size=150)

    return regressorGRU


def create_prophet_results(all_data):
    '''
    create prophet model trained on first 2768 rows by
    default and predicts on last 250 rows
    '''
    # Pull train data
    train_data = all_data[:-1].reset_index()[['Date', 'High']]
    train_data.columns = ['ds', 'y']

    # Create and fit model
    prophet_model = Prophet()
    prophet_model.fit(train_data)

    # Provide predictions
    test_dates = prophet_model.make_future_dataframe(periods=DAYS)
    forecast_prices = prophet_model.predict(test_dates)

    return forecast_prices


def predict_trend(model, X_test, sc):
    # predicted_prices = []
    # for i in range(0, DAYS):
    scaled_pred = predict_from_model(model, X_test)
    # X_test = np.append(X_test, scaled_pred)[1:].reshape(X_test.shape[0], X_test.shape[1], 1)
    # predicted_prices.append(sc.inverse_transform(scaled_pred))
    # print("confidence::", scaled_pred, "real value::", sc.inverse_transform(scaled_pred))
    return model, scaled_pred, sc.inverse_transform(scaled_pred)


def predict_trend_gru(model, X_test, sc):
    # predicted_prices = []
    # for i in range(0, DAYS):
    scaled_pred = predict_gru(model, X_test)
    # X_test = np.append(X_test, scaled_pred)[1:].reshape(X_test.shape[0], X_test.shape[1], 1)
    # predicted_prices.append(sc.inverse_transform(scaled_pred))
    # print("confidence::", scaled_pred, "real value::", sc.inverse_transform(scaled_pred))
    return model, scaled_pred, sc.inverse_transform(scaled_pred)


def plot_results(stock_data,
                 stock_name,
                 small_one_layer_preds,
                 one_layer_preds,
                 rnn_preds,
                 gru_preds,
                 gru_drop_preds,
                 most_confident_preds,
                 plot_pth='./figures'):
    '''
    plot the results
    '''
    plt.figure(figsize=(20, 5))

    historyData = stock_data["High"][-120:].values[:-DAYS]

    plt.plot(np.append(historyData, small_one_layer_preds), label='Single Layer Small RNN values', alpha=0.5)
    plt.plot(np.append(historyData, one_layer_preds), label='Single Layer RNN values', alpha=0.5)
    plt.plot(np.append(historyData, rnn_preds), label='RNN values', alpha=0.5)
    plt.plot(np.append(historyData, gru_preds), label='GRU without dropout values', alpha=0.5)
    plt.plot(np.append(historyData, gru_drop_preds), label='GRU with dropout values', alpha=0.5)
    plt.plot(np.append(historyData, most_confident_preds), label='Most confident preds', alpha=0.5)
    plt.plot(stock_data["High"][-120:].values[:], label='actual values', color='black', alpha=0.5)
    plt.title('{} Predictions vs. Actual'.format(stock_name))
    plt.legend()

    fig_path = os.path.join(plot_pth, 'results', 'mc_' + stock_name + '_preds')

    # save the data, pause, and close
    plt.savefig(fig_path)
    plt.close()


def process_data():
    all_data = create_files_dict("/home/nischit/Desktop/data/")
    i = 0
    for stock_name, stock_data in all_data.items():
        try:
            start = time.time() * 1000
            # if i > 50:
            #     print("Max run count reached")
            #     break
            print("#", i, "PROCESSING::", stock_name)
            # create dl data
            X_train, y_train, X_test, sc = create_dl_train_test_split(stock_data)
            if X_train is None:
                print("Skipping this because insufficient data")
                continue

            small_one_layer_preds = []
            one_layer_preds = []
            rnn_preds = []
            gru_preds = []
            gru_drop_preds = []
            predicted_prices = []

            single_layer_small_rnn_model = create_single_layer_small_rnn_model(X_train, y_train)
            single_layer_rnn_model = create_single_layer_rnn_model(X_train, y_train)
            rnn_model = create_rnn_model(X_train, y_train)
            GRU_model = create_GRU_model(X_train, y_train)
            GRU_with_drop_out_model = create_GRU_with_drop_out_model(X_train, y_train)

            for k in range(0, DAYS):
                print("processing prediction day #", k)
                small_single_layer_rnn, small_one_layer_scaled, small_one_layer_pred = predict_trend(
                    single_layer_small_rnn_model, X_test, sc)
                small_one_layer_preds.append(small_one_layer_pred[0][0])
                #
                # # create single layer rnn preds
                single_layer_rnn, one_layer_scaled, one_layer_pred = predict_trend(single_layer_rnn_model, X_test, sc)
                one_layer_preds.append(one_layer_pred[0][0])
                #
                # # rnn daily preds
                rnn_model, rnn_scaled, rnn_pred = predict_trend(rnn_model, X_test, sc)
                rnn_preds.append(rnn_pred[0][0])
                #
                # # gru daily preds
                gru_model, gru_scaled, gru_pred = predict_trend_gru(GRU_model, X_test, sc)
                gru_preds.append(gru_pred[0][0])
                #
                # # gru daily preds
                gru_drop_model, gru_drop_scaled, gru_drop_pred = predict_trend_gru(GRU_with_drop_out_model, X_test, sc)
                gru_drop_preds.append(gru_drop_pred[0][0])
                #
                most_confident = max(small_one_layer_scaled, one_layer_scaled, rnn_scaled, gru_scaled, gru_drop_scaled)
                predicted_prices.append(sc.inverse_transform(most_confident)[0][0])

                X_test = np.append(X_test, most_confident)[1:].reshape(X_test.shape[0], X_test.shape[1], 1)

            # plot results
            plot_results(stock_data,
                         stock_name,
                         small_one_layer_preds,
                         one_layer_preds,
                         rnn_preds,
                         gru_preds,
                         gru_drop_preds,
                         predicted_prices
                         )
            i += 1
            end = time.time() * 1000
            print("#", i, "runtime: ", (end - start) / 1000.0, "s")
        except:
            pass


if __name__ == '__main__':
    try:
        process_data()
    except:
        print("Something went wrong when processing data")
    print("DONE!")
