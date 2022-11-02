###########################################################
# Jim Moroney                                    10.27.22 #
# ParaARIMA.py                    Directed Studies in HPC #
# A program to evaluate an ARIMA model using a walk-forw- #
# ard validation.                                         #
###########################################################

# The current goal of this program is to rework Rajesh's
# arima.py, after that we shall look into parallelizing the
# auto arima process itself.

###########################################################
# library imports
import sys
import numpy as np
import pandas as pd
import datetime as dt
import pmdarima as pm 

from matplotlib import pyplot

###########################################################
# functions
def date_convert(date_to_convert): #To format date and time for ARIMA
     return dt.datetime.strptime(date_to_convert, "%m/%d/%y %H:%M").strftime("%Y-%m-%d")

###########################################################
# global variables (because python)
inputFile = "\0"
outputFile = "\0"
###########################################################
# command line argument handling
numArgs = len(sys.argv)
if numArgs <= 1:
    sys.exit("!!ERROR: Arguments expected, please run 'ParaARIMA.py -h' for help.\n")
#step through the arguments to parse them
for argIndex in range(1, numArgs): #start at 1 because argv[0] is just the name of the program.
    if sys.argv[argIndex] in ("-h", "--Help"):
        print ("########################################")
        print ("# ParaARIMA.py - ARIMA Parallelization #")
        print ("########################################")
        print ("[usage]")
        print ("\t[ -i : input ] \n\t\t*specifies the file to be read into memory")
        print ("\t[ -o : output ] \n\t\t*specifies the file to be written to file\n\t\t*if this option is not present the results will not be written to file")
        print ("\t[ -h : help ] \n\t\t*this! :)\n")
    elif sys.argv[argIndex] in ("-i", "--Input"):
        argIndex += 1
        inputFile = sys.argv[argIndex]
        print ("global inputFile set as: " + inputFile)
    elif sys.argv[argIndex] in ("-o", "--Output"):
        argIndex += 1
        outputFile = sys.argv[argIndex]
        print ("global outputFile set as: " + outputFile)

#######################################
# CSV Parsing and dataframe building
if inputFile == "\0":
    sys.exit("!!ERROR: No input file set, please run 'ParaArima.py -h' for help\n")

#read csv into a pandas dataframe
arimaFrame = pd.read_csv(inputFile, sep = ',', header = 0)
#remove last row
arimaFrame = arimaFrame.iloc[:-1]
#drop second date row
arimaFrame.drop(['date','Residential', 'Total'], axis = 1, inplace = True)
#rename the first column to Date, we can infer that the values are meter ID's
arimaFrame.rename(columns = {'AMI Meter ID':'Date'}, inplace = True)
#format date and time so pandas can recognize it as a date object
arimaFrame['Date'] = arimaFrame['Date'].str.replace(r'Eastern Standard Time','').str.rstrip().apply(date_convert)
arimaFrame.Date = pd.to_datetime(arimaFrame.Date)
print (arimaFrame)

st_arimaFrame = arimaFrame.set_index('Date')
#st_arimaFrame.columns = pd.to_timedelta(st_arimaFrame.columns + ':00')
st_arimaFrame = st_arimaFrame.stack()
#st_arimaFrame.index = st_arimaFrame.index.get_level_values(0) + st_arimaFrame.index.get_level_values(1)
st_arimaFrame = st_arimaFrame.reset_index()
st_arimaFrame.columns = ["Date", "AMI Meter ID", "Value"]
st_arimaFrame.astype({'AMI Meter ID':'int32'}, copy=False)

print (st_arimaFrame.head(5))
print (st_arimaFrame.info())

st_arimaFrame = st_arimaFrame.set_index('Date')

stVals = st_arimaFrame.values
print(stVals)
size = int(len(stVals) * 0.66)
train, test = stVals[0:size], stVals[size:len(stVals)]
history = [x for x in train]

history = np.delete(history, 0,1)

print(history)

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
pyplot.title("Actual vs Prediction for " + inputFile)
pyplot.plot(train,color='green',label='Training')
pyplot.plot(test, color='blue', label ='Test')
pyplot.plot(pr, color='red', label ='Predictions')
pyplot.legend(loc ='upper right')
pyplot.show()

