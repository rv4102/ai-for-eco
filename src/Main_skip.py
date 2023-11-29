from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import *
from model import *
MODEL_NAME = './stock_model_skip.h5'
SKIP = True
data1 = pd.read_csv("./datasets/nifty.csv")
data1.dropna(inplace=True)
data1.reset_index(inplace=True)
# print(len(data1))
# exit()
data1.index = pd.to_datetime(data1['Date'], format='%Y-%m-%d')
#data1 = data1.drop(['ts_code', 'Date', 'turnover_rate', 'volume_ratio', 'pb', 'total_share', 'float_share', 'free_share'], axis=1)
# data1 = data1.loc[:, ['open', 'high', 'low', 'Close', 'vol', 'amount']]
# data1 = data1.loc[:, ['Open', 'High', 'low', 'Close', 'vol', 'amount']]
data1 = data1.loc[:, ['Open', 'High', 'Low', 'Close']]
data_nifty = data1
residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['Date'])
residuals.pop('Date')
data1 = pd.merge(data1, residuals, on='Date')
# print(data1.head())
# exit()
# data = data1.iloc[1:3500, :]
data = data1.iloc[1:3501, :] 
# data = data1
data2 = data1.iloc[3501:, :] 
TIME_STEPS = 20
# print(data.head())
# print(data.iloc[0,3])
# exit()
data, normalize = NormalizeMult(data)
# print(data)
# exit()
# print('#', normalize)
pollution_data = data[:, 3].reshape(len(data), 1)

train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

print("SHAPE : ",train_X.shape, train_Y.shape)

# m = attention_model(INPUT_DIMS=5,TIME_STEPS=TIME_STEPS,skip_link=True)
# m.summary() 
# adam = Adam(learning_rate=0.01)
# m.compile(optimizer=adam, loss='mse') 
# history = m.fit([train_X], train_Y, epochs=50, batch_size=32, validation_split=0.1)
# m.save(MODEL_NAME)
# np.save("stock_normalize.npy", normalize)

# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# normalize = np.load("normalize.npy")
# loadmodelname = "model.h5"

class Config:
    def __init__(self):
        self.dimname = 'Close'

config = Config()
name = config.dimname
# normalize = np.load("normalize.npy")
print(len(data2))
y_hat, y_test = PredictWithDataAsli(data1, data_nifty, name, MODEL_NAME,5,skip_link=SKIP)
y_hat = np.array(y_hat, dtype='float64')
y_test = np.array(y_test, dtype='float64')
evaluation_metric(y_test,y_hat)
# time = pd.Series(data1.index[3499:])
time = pd.Series(data1.index[3522:])
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model with SKIP CONNECTION : prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()