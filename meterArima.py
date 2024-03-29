###########################################################
# Jim Moroney                                    10.27.22 #
# meterARIMA.py                   Directed Studies in HPC #
# A rewrite of ParaARIMA.py to treat the given data set   #
# as a set of independent meters.                         #
###########################################################

# The current goal of this program is to rework Rajesh's
# arima.py, after that we shall look into parallelizing the
# auto arima process itself.

###########################################################
# library imports
import os
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
import datetime as dt
# import pmdarima as pm 
from pmdarima.arima import ADFTest
from pmdarima.arima import auto_arima
from dataclasses import dataclass
from matplotlib import pyplot
from sklearn.metrics import r2_score

###########################################################
# Data Structures
@dataclass
class arimaData:
    isStationary = 0
    stationaryP = 0
    localTrain = 0
    localTest = 0
    localPrediction = 0
    arimaModel = 0
    r2Result = 0

@dataclass
class meterWrapper:
    meterID = 0 
    seasons = []
    models = []

frameCollection = [] # list for data frames
meterCollection = [] # to hold various data points for each arima model
processPool = []

###########################################################
# functions
def arima_process(meterIndex, seasonIndex, seasonNum, tempData):
    seasonPval, seasonStationary = adf_test.should_diff(seasonIndex)
    print("--season #", seasonNum, "(", seasonStationary, ",", seasonPval, ")")
    tempData.stationaryP = seasonPval
    tempData.isStationary = seasonStationary

    print("----splitting season into train and test...", end='')
    trainRows = int(len(seasonIndex) * trainVal) #Get the number of rows that equals the training percentage
    tempData.localTrain = seasonIndex[:trainRows] #put those rows into a variable
    tempData.localTest = seasonIndex.drop(tempData.localTrain.index) #throw whats left into test
    
    print("Done!")

    print("----calculating auto_arima...")
    if tempData.isStationary == True:
        tempData.arimaModel = auto_arima(tempData.localTrain, start_p=1,start_q=1,
                                    test='adf',
                                    max_p=10,max_q=10,
                                    m=1,
                                    d=None, start_P=0,
                                    D=None,
                                    seasonal=True,
                                    error_action='warn',trace=True,
                                    supress_warnings=True,stepwise=True)
    elif tempData.isStationary == False:
        tempData.arimaModel = auto_arima(tempData.localTrain, start_p=1,start_q=1,
                                    test='adf',
                                    max_p=10,max_q=10,
                                    m=1,
                                    d=None, start_P=0,
                                    D=None,
                                    seasonal=True,
                                    error_action='warn',trace=True,
                                    supress_warnings=True,stepwise=True)
    print("----Done!")
    print("----calculating prediction...", end='')
    tempData.localPrediction = pd.DataFrame(tempData.arimaModel.predict(n_periods=len(tempData.localTest),
                                                                        index=tempData.localTest.index))
    tempData.localPrediction.columns = ['predicted']
    print("Done!")
    print("----calculating r2 score...", end='')
    #r2 score calculation
    tempData.r2Result = r2_score(tempData.localTest, tempData.localPrediction)
    print("Done!")
    print("r2 score:", tempData.r2Result)
    
    pyplot.plot(tempData.localTrain, label = "Training")
    pyplot.plot(tempData.localTest, label = "Test")
    pyplot.plot(tempData.localPrediction, label = "Predicted")
    pyplot.legend(loc = 'upper left')
    pyplot.title(str(meterIndex.meterID) + ", season " + str(seasonNum) + ", r2:" + str(tempData.r2Result))

    pyplot.savefig(outputFile+str(meterIndex.meterID)+"_season"+str(seasonNum)+".png", dpi=300)
    pyplot.close()

    meterIndex.models.append(tempData)


###########################################################
# global variables (because python)
adf_test = ADFTest(alpha = 0.05)
inputFile = "\0"
outputFile = "./"
seasonality = 'M'
execution = "s"
trainVal = .8


if __name__ == "__main__":

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
            print ("\t[ -i : --Input ] \n\t\t*specifies the file to be read into memory")
            print ("\t[ -o : --Output ] \n\t\t*specifies the file to be written to\n\t\t*if this option is not present the results will not\n\t\t*be written to file")
            print ("\t[ -s : --Season ] \n\t\t*specifies the seasonality of how the program will\n\t\t*split the given data for modeling\n\t\t*d - daily, w - weekly, m - monthly, q - quarterly, b - biyearly\n\t\t*default is monthly\n\t\t!!WARNING: See documentation for possible complications")
            print ("\t[ -e : --Execute ] \n\t\t*specifies whether the arima calculations will run\n\t\t*in serial (s) or parallel (p)\n\t\t!!WARNING: See documentation for possible complications")
            print ("\t[ -h : --Help ] \n\t\t*this! :)\n")
        elif sys.argv[argIndex] in ("-i", "--Input"):
            argIndex += 1
            inputFile = sys.argv[argIndex]
            print ("global inputFile set as: " + inputFile)
        elif sys.argv[argIndex] in ("-o", "--Output"):
            argIndex += 1
            outputFile = sys.argv[argIndex]
            print ("global outputFile set as: " + outputFile)
        elif sys.argv[argIndex] in ("-s", "--Season"):
            argIndex += 1
            if sys.argv[argIndex].upper() in ('D', 'W', 'M'):
                seasonality = sys.argv[argIndex].upper()
            elif sys.argv[argIndex].upper() == 'Q':
                seasonality = '3M'
            elif sys.argv[argIndex].upper() == 'B':
                seasonality = '6M'
            else:
                sys.exit("!!ERROR: Invalid seasonality, please run 'paraARIMA.py -h for help\n") 
            print("global seasonality set as: " + seasonality)
        elif sys.argv[argIndex] in ("-e", "--Execute"):
            argIndex += 1
            if sys.argv[argIndex] in ('s','p'):
                execution = sys.argv[argIndex]
            else:
                sys.exit("!!ERROR: Invalid execution mode, please run 'paraARIMA.py -h for help\n")


        # else:
        #     sys.exit("!!ERROR: Argument \'" + sys.argv[argIndex] + "\' not recognized, please run 'ParaARIMA.py -h' for help.\n")




    #######################################
    # CSV Parsing and dataframe building
    if inputFile == "\0":
        sys.exit("!!ERROR: No input file set, please run 'ParaArima.py -h' for help\n")


    StartTime = time.time()
    #read csv into a pandas dataframe
    arimaFrame = pd.read_csv(inputFile, sep = ',', header = 0)

    #this data conditionining is very specific to the dataset
    #remove last row
    arimaFrame = arimaFrame.iloc[:-1]
    #drop second date row
    arimaFrame.drop(['date','Residential', 'Total'], axis = 1, inplace = True)
    #rename the first column to Date, we can infer that the values are meter ID's
    arimaFrame.rename(columns = {'AMI Meter ID':'Date'}, inplace = True)
    arimaFrame.ffill(inplace=True)
    arimaFrame['Date'] = pd.to_datetime(arimaFrame['Date'])
    EndTime = time.time()

    print(arimaFrame.head(5))

    print("\033[93m!!TIMING: CSV Input done in {:.4f} seconds \033[0m".format(EndTime-StartTime))

    # split CSV into as many dataframes as there are columns

    StartTime = time.time()
    for dfIndex, dfColumns in enumerate(arimaFrame.columns[1:]): #skip the first column because its date
        print("Processing column: ", dfIndex+1, " ", dfColumns)
        tempFrame = arimaFrame.iloc[:, [0, dfIndex+1]].copy()
        tempFrame.set_index('Date', inplace = True)
        frameCollection.append(tempFrame)
    EndTime = time.time()

    print("\033[93m!!TIMING: CSV Split done in {:.4f} seconds \033[0m".format(EndTime-StartTime))

    ###########################################################
    # ARIMA Preconditioning

    #Splitting frames into seasons
    StartTime = time.time()
    for frameIndex in frameCollection:
        tempID = frameIndex.columns[0]
        print("Splitting meter " + tempID + " into seasons...", end='')
        tempMeter = meterWrapper()
        tempMeter.meterID = tempID

        tempGroups = frameIndex.groupby(pd.Grouper(freq=seasonality)) # Binning the data by seasonality, noted in documentation can be more values than whats suggested
                                                            # TO DO: logic for various seasonality inputs
        tempMeter.seasons = [group for groupIndex, group in tempGroups]
        meterCollection.append(tempMeter)
        print("Done!")
    EndTime = time.time()

    print("\033[93m!!TIMING: Season Split done in {:.4f} seconds \033[0m".format(EndTime-StartTime))

    ###########################################################
    # Arima Calculations
    if execution == 's':
        print("\033[92m Serial ARIMA Calculation selected...\033[0m")
        meterNum = 0
        for meterIndex in meterCollection:
            meterNum += 1
            seasonNum = 0
            print("ADF on #", meterNum, ":", meterIndex.meterID)
            StartTime = time.time()
            for seasonIndex in meterIndex.seasons:
                seasonNum += 1
                tempData = arimaData()
                arima_process(meterIndex, seasonIndex, seasonNum, tempData)
            EndTime = time.time()
            print("\033[93m!!TIMING: ARIMA processing done in {:.4f} seconds \033[0m".format(EndTime-StartTime))
    elif execution == 'p':
        print("\033[92m Parallel ARIMA Calculation selected...\033[0m")
        multiprocessing.set_start_method('spawn')
        meterNum = 0
        for meterIndex in meterCollection:
            meterNum += 1
            seasonNum = 0
            print("ADF on #", meterNum, ":", meterIndex.meterID)
            StartTime = time.time()
            for seasonIndex in meterIndex.seasons:
                seasonNum += 1
                tempData = arimaData()
                proc = multiprocessing.Process(target=arima_process, args=(meterIndex, seasonIndex, seasonNum, tempData,))
                proc.start()
                processPool.append(proc)
            for process in processPool:
                process.join()
            EndTime = time.time()
            print("\033[93m!!TIMING: ARIMA processing done in {:.4f} seconds \033[0m".format(EndTime-StartTime))
    else:
        sys.exit("\033[91m!!ERROR: Unable to determine execution mode. \033[0m")   
    #pyplot.show()