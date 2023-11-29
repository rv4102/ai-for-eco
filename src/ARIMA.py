import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics
from utils import *
#plt.rcParams['font.sans-serif'] = ['SimHei']    # for chinese text on plt
#plt.rcParams['axes.unicode_minus'] = False      # for chinese text negative symbol '-' on plt
P = 2
D = 1
Q = 4
# data = pd.read_csv('./601988.SH.csv')
data = pd.read_csv('./datasets/nifty.csv')
# print(len(data))
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
print(data.iloc[3500,0])
# print(len(data))
test_set2 = data.loc[3501:, :] 
train_end_date = data.loc[3501, 'Date']
print(train_end_date)
# test_start_date = data.loc[3501,'Date']
print(test_set2.head())
data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d') 
data = data.drop(['Date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)
print(len(data))
training_set = data.loc[:train_end_date, :]  # 3501
test_set = data.loc[train_end_date:, :]  # 180
print(len(test_set),len(test_set2))
plt.figure(figsize=(10, 6))
plt.plot(training_set['Close'], label='training_set')
plt.plot(test_set['Close'], label='test_set')
plt.title('Close price')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

temp = np.array(training_set['Close'])

# First-order diff
training_set['diff_1'] = training_set['Close'].diff(1)
plt.figure(figsize=(10, 6))
training_set['diff_1'].plot()
plt.title('First-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.show()

# Second-order diff
training_set['diff_2'] = training_set['diff_1'].diff(1)
plt.figure(figsize=(10, 6))
training_set['diff_2'].plot()
plt.title('Second-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_2', fontsize=14, horizontalalignment='center')
plt.show()

temp1 = np.diff(training_set['Close'], n=1)

# white noise test
training_data1 = training_set['Close'].diff(1)
# training_data1_nona = training_data1.dropna()
temp2 = np.diff(training_set['Close'], n=1)
# print(acorr_ljungbox(training_data1_nona, lags=2, boxpierce=True, return_df=True))
print(acorr_ljungbox(temp2, lags=2, boxpierce=True))
# p-value=1.53291527e-08, non-white noise time-seriess
print(training_set['Close'], type(training_set['Close']))
acf_pacf_plot(training_set['Close'],acf_lags=250)
# acf_pacf_plot(training_set['Close'],acf_lags=250)
# acf_pacf_plot(training_data_diff)
price = list(temp2)
data2 = {
    'Date': training_set['diff_1'].index[1:], 
    'Close': price
}

df = pd.DataFrame(data2)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# print(test_set2.head())
# print(test_set2.iloc[0,4])
# exit()
training_data_diff = df.set_index(['Date'], drop=True)
print('&', training_data_diff,len(training_data_diff))
# print('&&', training_data_diff.index)
acf_pacf_plot(training_data_diff,40,40)
# exit()
# acf_pacf_plot(training_data_diff,100)
print("CHECK : ",test_set.shape[0],len(test_set2))
# order_select_ic(training_data_diff)
# order_select_search(training_set)
# order=(p,d,q)
# p and q --> 2,4
model = sm.tsa.ARIMA(endog=training_set['Close'], order=(P, D, Q)).fit()
#print(model.summary())

history = [x for x in training_set['Close']]
# print('history', type(history), history)
predictions = list()
# print('test_set.shape', test_set.shape[0])
for t in range(test_set.shape[0]):
    model1 = sm.tsa.ARIMA(history, order=(P, D, Q))
    model_fit = model1.fit()
    yhat = model_fit.forecast()
    # print("yhat : ",yhat,len(yhat))
    yhat = np.float(yhat[0])
    predictions.append(yhat)
    # print(type(test_set2['Close']),len(test_set2['Close']))
    obs = test_set2.iloc[t,4]
    # obs = test_set2['Close'].to_list()[t]
    # obs = np.float(obs)
    # print('obs', type(obs))
    history.append(obs)
    # print(test_set.index[t])
    # print(t+1, 'predicted=%f, expected=%f' % (yhat, obs))
#print('predictions', predictions)

predictions1 = {
    'Date': test_set.index[:],
    'Close': predictions
}
predictions1 = pd.DataFrame(predictions1)
predictions1 = predictions1.set_index(['Date'], drop=True)
predictions1.to_csv('./ARIMA.csv')
plt.figure(figsize=(10, 6))
plt.plot(test_set['Close'], label='Stock Price')
plt.plot(predictions1, label='Predicted Stock Price')
plt.title('ARIMA: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
# print(data.iloc[3501,3])
model2 = sm.tsa.ARIMA(endog=data['Close'], order=(P, D, Q)).fit()
residuals = pd.DataFrame(model2.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
residuals.to_csv('./ARIMA_residuals1.csv')
evaluation_metric(test_set['Close'],predictions)
adf_test(temp)
adf_test(temp1)

predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
predictions_ARIMA_diff = predictions_ARIMA_diff[3479:]
print('#', predictions_ARIMA_diff)
plt.figure(figsize=(10, 6))
plt.plot(training_data_diff, label="diff_1")
plt.plot(predictions_ARIMA_diff, label="prediction_diff_1")
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.title('DiffFit')
plt.legend()
plt.show()