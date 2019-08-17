########################################################################################################################
# Code to check that no sorts snuck into the data and export the final file
#       This code exists as there is no primary key. Initially I naively recombined my sets w/o resorting on the original
#       sequence number.
#   Steps:  0: Header + readin Files
#           1: Test that the AUC is not 50% due to changes in order
#           2: Create final files.
########################################################################################################################

##### Step 0: Header #####

import pandas as pd
import numpy as np

TGBM = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/TrainGBM.csv')
TGBM.shape
TGBM.head()
TGBM.tail()

TKNN = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/KNNTrain.csv')
TKNN.shape
TKNN.head()
TKNN.tail()


HOGBM = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/HOGBM.csv')
print(HOGBM.head())
print(HOGBM.shape)
print(HOGBM.tail())

HOKNN = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/KNNHO.csv')
print(HOKNN.head())
print(HOKNN.shape)
print(HOKNN.tail())

##### Step 1: the following quick check prevented a major error... ######
from sklearn.metrics import roc_auc_score as AUC
Raw = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/RAWDATA/exercise_05_train.csv')
Raw['y'][1:10]

print(AUC(Raw['y'], TGBM['PredictGBM']))


##### step 2: Export file ######
TExport = pd.concat([TGBM, TKNN], axis= 1)
TExport.shape

TExport.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/TrainingScores.csv',
               index = False,
               header = None)


HOExport = pd.concat([HOGBM, HOKNN], axis= 1)
HOExport.shape
HOExport.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/HoldOutScores.csv',
               index = False,
               header = None)


HOKNN['PredictKNN'] = HOKNN['PredictKNN'].astype(float)

HOKNN.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Code/FinalPacket/results1.csv',
               index = False,
               header = None)

HOGBM.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Code/FinalPacket/results2.csv',
               index = False,
               header = None)

pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Code/FinalPacket/results1.csv').head()
pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Code/FinalPacket/results2.csv').head()
