import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam
# from tensorflow.keras.optimizers import Adam
from numpy.random import seed
from utils import *
from model import lstm

# GPU
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#     tf.config.set_visible_devices([gpus[0]], "GPU")

seed(1)
tf.random.set_seed(1)

n_timestamp = 10
n_epochs = 50
# ====================================
#      model type：
#            1. single-layer LSTM
#            2. multi-layer LSTM
#            3. bidirectional LSTM
# ====================================
model_type = 3

nifty_data = pd.read_csv('./datasets/nifty.csv') 
nifty_data.dropna(inplace=True) 
nifty_data.reset_index(inplace=True,drop=True)
nifty_data.index = pd.to_datetime(nifty_data['Date'], format='%Y-%m-%d') 
idx = 3501
end_date = nifty_data.iloc[idx,0] 
print("CHECK : ",end_date)  
nifty_data = nifty_data.loc[:, ['Open', 'High', 'Low', 'Close']]

data = pd.read_csv('./ARIMA_residuals1.csv')
data.index = pd.to_datetime(data['Date'])
print(len(nifty_data), len(data))
# exit()
# print(data.head())
data = data.drop('Date', axis=1)
# data = pd.merge(data, nifty_data, on='Date') 

Lt = pd.read_csv('./ARIMA.csv')
training_set = data.iloc[1:idx, :]
test_set = data.iloc[idx:, :]
nifty_training_set = nifty_data.iloc[1:idx, :]
nifty_test_set = nifty_data.iloc[idx:, :]

sc = MinMaxScaler(feature_range=(0, 1))
nifty_sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(test_set)
nifty_training_set_scaled = nifty_sc.fit_transform(nifty_training_set)
nifty_testing_set_scaled = nifty_sc.fit_transform(nifty_test_set)

X_train, y_train = data_split(training_set_scaled, n_timestamp)
nifty_X_train, nifty_y_train = data_split(nifty_training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
nifty_X_train = nifty_X_train.reshape(nifty_X_train.shape[0], nifty_X_train.shape[1], 4)

X_test, y_test = data_split(testing_set_scaled, n_timestamp)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
nifty_X_test, nifty_y_test = data_split(nifty_testing_set_scaled, n_timestamp)
yuna_X_test = nifty_X_test.reshape(nifty_X_test.shape[0], nifty_X_test.shape[1], 4)

model, nifty_model = lstm(model_type,X_train,nifty_X_train)
print(model.summary())
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam,
              loss='mse')
nifty_model.compile(optimizer=adam,
                   loss='mse')

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),
                    validation_freq=1)
nifty_history = nifty_model.fit(nifty_X_train, nifty_y_train,
                              batch_size=32,
                              epochs=n_epochs,
                              validation_data=(nifty_X_test, nifty_y_test),
                              validation_freq=1)


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('residuals: Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(nifty_history.history['loss'], label='Training Loss')
plt.plot(nifty_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM: Training and Validation Loss')
plt.legend()
plt.show()

nifty_predicted_stock_price = nifty_model.predict(nifty_X_test)
nifty_predicted_stock_price = nifty_sc.inverse_transform(nifty_predicted_stock_price)
nifty_predicted_stock_price_list = np.array(nifty_predicted_stock_price[:, 3]).flatten().tolist()
nifty_predicted_stock_price1 = {
    'Date': nifty_data.index[idx+10:],
    'Close': nifty_predicted_stock_price_list
}
nifty_predicted_stock_price1 = pd.DataFrame(nifty_predicted_stock_price1)
nifty_predicted_stock_price1 = nifty_predicted_stock_price1.set_index(['Date'], drop=True)
nifty_real_stock_price = nifty_sc.inverse_transform(nifty_y_test)
nifty_real_stock_price_list = np.array(nifty_real_stock_price[:, 3]).flatten().tolist()
nifty_real_stock_price1 = {
    'Date': nifty_data.index[idx+10:],
    'Close': nifty_real_stock_price_list
}
nifty_real_stock_price1 = pd.DataFrame(nifty_real_stock_price1)
nifty_real_stock_price1 = nifty_real_stock_price1.set_index(['Date'], drop=True)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price_list = np.array(predicted_stock_price[:, 0]).flatten().tolist()

predicted_stock_price1 = {
    'Date': data.index[idx+10:],
    'Close': predicted_stock_price_list
}
predicted_stock_price1 = pd.DataFrame(predicted_stock_price1)

predicted_stock_price1 = predicted_stock_price1.set_index(['Date'], drop=True)

real_stock_price = sc.inverse_transform(y_test)
finalpredicted_stock_price = pd.concat([Lt, predicted_stock_price1]).groupby('Date')['Close'].sum().reset_index()
finalpredicted_stock_price.index = pd.to_datetime(finalpredicted_stock_price['Date']) # 将时间格式改变一下
finalpredicted_stock_price = finalpredicted_stock_price.drop(['Date'], axis=1)

plt.figure(figsize=(10, 6))
# print('nifty_real', nifty_real_stock_price1)
plt.plot(nifty_data.loc[end_date:, 'Close'], label='Stock Price')
plt.plot(finalpredicted_stock_price['Close'], label='Predicted Stock Price')
plt.title('BiLSTM: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(nifty_real_stock_price1['Close'], label='Stock Price')
plt.plot(nifty_predicted_stock_price1['Close'], label='Predicted Stock Price')
plt.title('LSTM: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

yhat = nifty_data.loc[end_date:, 'Close']
# print(finalpredicted_stock_price.iloc[30:,:].head())
# print(nifty_data.loc[end_date:,:].head())

evaluation_metric(finalpredicted_stock_price['Close'],yhat)

np.save('lstm_pred.npy', finalpredicted_stock_price['Close'].values)
np.save('real.npy', yhat.values)
# save finalpredicted_stock_price

