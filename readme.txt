paraArima.py
A program for calculating arima models in parallel.

Jim Moroney, 2022
Tennessee Technological University

Last Update: 12.10.2022

QUICKSTART

python3 paraArima.py -i data.csv

REQUIREMENTS
-Data
paraArima.py works under the assumption that the data given to it is of the form provided in "AllAssetData.csv." That is to say: a collection of independent meters, keyed by their meter ID's and with readings indexed by the time at which they were taken.

In its current state, the data conditioning that paraArima.py completes before processing is static, and is unable to be defined by the user (see planned features). 

COMMAND LINE ARGUMENTS

TECHNICAL OUTLINE
This section is a rough outline describing the operation of the program as it executes, 

PLANNED FEATURES
Data config file
season averaging 
message passing
