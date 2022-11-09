###########################################################
# Jim Moroney                                    10.27.22 #
# indARIMA.py                     Directed Studies in HPC #
# A reqrite of ParaARIMA.py to treat the given data set   #
# as a set of independent meters.                         #
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
# import pmdarima as pm 
from pmdarima.arima import ADFTest

from matplotlib import pyplot

###########################################################
# global variables (because python)
inputFile = "\0"
outputFile = "\0"
frameCollection = [] # array for data frames 
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

#this data conditionining is very specific to the dataset we
#remove last row
arimaFrame = arimaFrame.iloc[:-1]
#drop second date row
arimaFrame.drop(['date','Residential', 'Total'], axis = 1, inplace = True)
#rename the first column to Date, we can infer that the values are meter ID's
arimaFrame.rename(columns = {'AMI Meter ID':'Date'}, inplace = True)
arimaFrame.ffill(inplace=True)
arimaFrame['Date'] = pd.to_datetime(arimaFrame['Date'])

print(arimaFrame.head(5))

# split CSV into as many dataframes as there are columns

for dfIndex, dfColumns in enumerate(arimaFrame.columns[1:]): #skip the first column because its date
    print("Processing column: ", dfIndex+1, " ", dfColumns)
    tempFrame = arimaFrame.iloc[:, [0, dfIndex+1]].copy()
    tempFrame.set_index('Date', inplace = True)
    frameCollection.append(tempFrame)

for frameIndex in frameCollection:
#     print(frameIndex)
    frameIndex.plot()

adf_test = ADFTest(alpha = 0.05)

meterIndex = 0
for frameIndex in frameCollection:
    meterIndex += 1
    print("ADF on #", meterIndex, ":", frameIndex.columns, adf_test.should_diff(frameIndex))
    
pyplot.show()