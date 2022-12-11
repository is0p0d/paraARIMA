# evaluate an ARIMA model using a walk-forward validation
from cProfile import label
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from math import sqrt
import pandas as pd
import os
from numpy import log
import numpy as np
import datetime as dt

#https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
#https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
#https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd
##############################DR for feb 2022#####################################################
#'400214601','400598000','901510','712010' 

#format date and time
def date_convert(date_to_convert):
     return dt.datetime.strptime(date_to_convert, "%A, %Y %B %d %H:%M:%S").strftime("%Y-%m-%d")

variable = '400214601'
cwd = os.getcwd() + "\\CustBehaviour\\data\\DR-Feb\\400214601.csv"
df=pd.read_csv(cwd,sep=',',header =0)
df = df.iloc[:-1] #remove last row Totals by Interval
df.drop(['Begin Read','End Read','Reg Usage','Interval Totals','Reg Diff'],axis=1, inplace=True)
df = df.fillna(method ='ffill')
df['Date'] = df['Date'].str.replace(r'Eastern Standard Time','').str.rstrip().apply(date_convert)
df.Date = pd.to_datetime(df.Date) 

df=df.set_index('Date')
df.columns = pd.to_timedelta(df.columns + ':00')
df = df.stack()
df.index = df.index.get_level_values(0) + df.index.get_level_values(1)
df = df.reset_index()
df.columns = ['Date','val']
print(df.head(5))
print(df.info())
df=df.set_index('Date')
df.plot()

#split train and test data
X = df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]



model = pm.auto_arima(history, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
n_periods = 672

#, index=test.index)
pr= pd.DataFrame(model.predict(n_periods=n_periods))
pr.columns = ['Predicted Consumption']
print(pr)

pyplot.figure(figsize=(8,5))
pyplot.title("Actual vs Prediction for " + variable )
pyplot.plot(train,color='green',label='Training')
pyplot.plot(test, color='blue', label ='Test')
pyplot.plot(pr, color='red', label ='Predictions')
pyplot.legend(loc ='upper right')
pyplot.show()
#pyplot.savefig(variable +'.png')