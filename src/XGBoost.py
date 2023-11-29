import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import walk_forward_validation

data = pd.read_csv('./datasets/nifty.csv')
data.dropna(inplace=True)
data.reset_index(inplace=True,drop=True)
train_end_date = data.loc[3501, 'Date']
data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')
print(train_end_date)
data = data.loc[:, ['Open', 'High', 'Low', 'Close']]
# data = pd.DataFrame(data, dtype=np.float64)
# Close = data.pop('Close')
# data.insert(3, 'Close', Close)
# print(data.head())
# exit()
data1 = data.iloc[3501:, 3]


residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['Date'])
residuals.pop('Date')
merge_data = pd.merge(data, residuals, on='Date')
#merge_data = merge_data.drop(labels='2007-01-04', axis=0)
time = pd.Series(data.index[3501:])
print(time.head())
Lt = pd.read_csv('./ARIMA.csv')
print(Lt.head())
Lt = Lt.drop('Date', axis=1)
Lt = np.array(Lt)
Lt = Lt.flatten().tolist()
# print(time.head())
train, test = prepare_data(merge_data, n_test=180, n_in=6, n_out=1)
print(train.head(),test.head())
# exit()
y, yhat = walk_forward_validation(train, test)
print("CHECK : ",len(y),len(yhat),len(time))
plt.figure(figsize=(10, 6))
plt.plot(time, y, label='Residuals')
plt.plot(time, yhat, label='Predicted Residuals')
plt.title('ARIMA+XGBoost: Residuals Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

finalpredicted_stock_price = [i + j for i, j in zip(Lt, yhat)]
#print('final', finalpredicted_stock_price)
evaluation_metric(data1, finalpredicted_stock_price)
plt.figure(figsize=(10, 6))
plt.plot(time, data1, label='Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
plt.title('ARIMA+XGBoost: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
