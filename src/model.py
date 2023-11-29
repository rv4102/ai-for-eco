from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Concatenate, Flatten
from keras.models import Model
import keras.backend as K
# from attention_utils import get_activations
# from keras.layers import merge
from keras.layers import Multiply
from keras.layers.core import *
from keras.layers import LSTM
from keras.models import *
from src.utils import *
import numpy as np
import xgboost as xgb

def attention_3d_block_merge(inputs,single_attention_vector = False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Multiply([inputs, a_probs],name='attention_mul')
    return output_attention_mul

def attention_3d_block(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(INPUT_DIMS = 13,TIME_STEPS = 20,lstm_units = 64,skip_link = False):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # padding = 'same'
    x = Dropout(0.3)(x)

    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    print(attention_mul.shape) # changed
    if skip_link:
        attention_mul = Concatenate()([attention_mul, Flatten()(lstm_out)])
    print("New shape", attention_mul.shape) # changed
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def PredictWithData(data,data_nifty,name,modelname,INPUT_DIMS = 13,TIME_STEPS = 20,do_normalize = True,skip_link = False):
    print(data.columns)
    print(data_nifty.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    if do_normalize:
        data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _ = create_dataset(data)
    _, testY = create_dataset(data_y)
    print("testX Y shape is:", testX.shape, testY.shape)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)
    model = attention_model(INPUT_DIMS,skip_link = skip_link)
    model.load_weights(modelname)
    model.summary()
    y_hat =  model.predict(testX)
    print("y_hat here", y_hat)
    testY, y_hat = xgb_scheduler(data_nifty, y_hat)
    return y_hat, testY

def PredictWithDataAsli(data,data_nifty,name,modelname,INPUT_DIMS = 13,TIME_STEPS = 20,skip_link = False):
    print(data.columns)
    print(data_nifty.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _ = create_dataset(data)
    _, testY = create_dataset(data_y)
    print("testX Y shape is:", testX.shape, testY.shape)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)

    model = attention_model(INPUT_DIMS,skip_link = skip_link)
    model.load_weights(modelname)
    model.summary()
    y_hat =  model.predict(testX)
    normalize = np.load("stock_normalize.npy")
    # y_hat = FNormalizeMult(y_hat, normalize[yindex])
    # testY = FNormalizeMult(testY, normalize[yindex])
    # print("y_hat here", y_hat)
    data_nifty = data_nifty.iloc[21:, :]
    data_nifty['y_hat'] = y_hat
    testY, y_hat = xgb_scheduler_asli(data_nifty, y_hat)
    return y_hat, testY

def lstm(model_type,X_train,nifty_X_train):
    if model_type == 1:
        # single-layer LSTM
        model = Sequential()
        model.add(LSTM(units=50, activation='relu',
                    input_shape=(X_train.shape[1], 1)))
        model.add(Dense(units=1))
        nifty_model = Sequential()
        nifty_model.add(LSTM(units=50, activation='relu',
                    input_shape=(nifty_X_train.shape[1], 4)))
        nifty_model.add(Dense(units=4))
    if model_type == 2:
        # multi-layer LSTM
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(1))

        nifty_model = Sequential()
        nifty_model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(nifty_X_train.shape[1], 4)))
        nifty_model.add(LSTM(units=50, activation='relu'))
        nifty_model.add(Dense(4))
    if model_type == 3:
        # BiLSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'),
                                input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))

        nifty_model = Sequential()
        nifty_model.add(Bidirectional(LSTM(50, activation='relu'),
                                    input_shape=(nifty_X_train.shape[1], 4)))
        nifty_model.add(Dense(4))

    return model,nifty_model

def xgb_scheduler(data,y_hat):
    Close = data.pop('Close')
    # data.insert(5, 'Close', Close)
    data.insert(3, 'Close', Close)
    # train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    print("Train: ", train)
    print("Test: ", test)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2

def xgb_scheduler_asli(data,y_hat):
    Close = data.pop('Close')
    # data.insert(5, 'Close', Close)
    data.insert(4, 'Close', Close)
    # train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    print("Train: ", train)
    print("Test: ", test)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2

def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # print('train', train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # print('trainX', trainX, 'trainy', trainy)
    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def walk_forward_validation(train, test):
    predictions = list()
    train = train.values
    history = [x for x in train]
    # print('history', history)
    for i in range(len(test)):
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        # print('i', i, testX, testy)
        yhat = xgboost_forecast(history, testX)
        # print("here: ", yhat)
        predictions.append(yhat)
        history.append(test.iloc[i, :])
        print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
    return test.iloc[:, -1],predictions