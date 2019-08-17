########################################################################################################################
# Code to solve state farms interview- Question 1.-build a K nearest neighbor model:
#
# Remarks:
#   Preliminary variable processing done in StateFarmExercisizeLoadData.py
#   Basic variable selection begun in VariableEDA.ipynb
#
# Step 0: Header & load data
#      1: Variable Processing and PCA
#      2: Final Model with a test set- 4 nearest neighbors selected based on AUC
#      3: KNN using 5 neighbors
#      4: Final Scorign
#
########################################################################################################################

##### Step 0: Header #####

import os
import gc
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

Train1 = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Train1.csv', sep = '|')
Train2 = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Train2.csv', sep = '|')
Test = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Test.csv', sep = '|')


def Tweaks(df):
    # minor data changes
    df['one'] = 1

    df.rename(columns={'Unnamed: 0': 'OriginalSeqNumber'}, inplace=True)

    # collapsing thin days with adjacent days
    df['dayInd1'] = ((df['x35rec'] == 'monday' ) | (Train1['x35rec'] == 'tuesday')) + 0
    df['dayInd2'] = ((df['x35rec'] == 'thurday' ) | (Train1['x35rec'] == 'tuesday') | (Train1['x35rec'] == 'friday'))+ 0
    df['dayInd3'] = (df['x35rec'] == 'wednesday' )  + 0

    return df

Train1 = Tweaks(Train1)
Train2 = Tweaks(Train2)
Test = Tweaks(Test)


T1y = Train1['y']
T2y = Train2['y']
Testy = Test['y']

BetterThanRandom = ['x37', 'x58', 'x75', 'x97', 'x41_rec_num', 'x70', 'x51', 'x2', 'x66', 'x96', 'x56',
                    'x1', 'x78', 'x79', 'x99', 'x40', 'x73', 'x72', 'x63', 'x5', 'x22', 'x83', 'x69', 'x33',
                    'x3', 'x45_rec_num', 'x50', 'x85', 'x44', 'x21', 'x20', 'x10', 'x0', 'x68SummerIndicator',
                    'x68FallIndicator', 'x38', 'x8', 'x74', 'x92', 'x53', 'x48', 'x29', 'x67', 'x95', 'x80',
                    #'x68SpringIndicator', delibrately left out to implicitly combine with the tiny winter sample, maybe combine with Fall?
                    'x52','dayInd1', 'dayInd2', 'dayInd3'
                    ]

Train1X = Train1[BetterThanRandom]
Train2X = Train2[BetterThanRandom]
TestX = Test[BetterThanRandom]

##### Step1: Variable processing #####

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder as OneHot

Train1Recaler = StandardScaler()
Train1Recaler.fit_transform(Train1X)
Train1Recaled = pd.DataFrame(Train1Recaler.transform(Train1X))
Train2Recaled = pd.DataFrame(Train1Recaler.transform(Train2X))

pca = PCA(n_components=45)
pca.fit_transform(Train1Recaled)

PrinCompTrain1 = pd.DataFrame(pca.transform(Train1Recaled))
PrinCompTrain2 = pd.DataFrame(pca.transform(Train2Recaled))

##### Step1: build KNN model #####

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score as AUC

for n_neighbors in range(2, 15):
    KNNModel = KNeighborsClassifier(n_neighbors= n_neighbors, weights = 'distance')
    KNNModel.fit(PrinCompTrain1, T1y)
    #Train1['Predict'] = KNNModel.predict(PrinCompTrain1)
    Train2['Predict'] = KNNModel.predict(PrinCompTrain2)
    print(n_neighbors)
    print(AUC(T2y, Train2['Predict']))

#6 seems best: similar all are similar tho (Shockingly :p). don't go over 10

##### Step2: Final Model with a test set ######
# : use KNN on Full set. Validataion Statistics will not be meaningful#####

#Train2.drop(columns='Predict', inplace=True)

TrainFullX = pd.concat([Train1X, Train2X]) #, TestX
TrainFully = pd.concat([T1y, T2y], axis= 0) #Testy

TrainRecaler = StandardScaler()
TrainRecaler.fit_transform(TrainFullX)
TrainRecaled = pd.DataFrame(Train1Recaler.transform(TrainFullX))
pca = PCA(n_components=45)
pca.fit_transform(TrainRecaled)
PrinCompTrainFull = pd.DataFrame(pca.transform(TrainRecaled))
TestRecaled = pd.DataFrame(Train1Recaler.transform(TestX))
PrinCompTest = pd.DataFrame(pca.transform(TestRecaled))

for n_neighbors in range(2, 10):
    KNNModel = KNeighborsClassifier(n_neighbors= n_neighbors, weights = 'distance')
    KNNModel.fit(PrinCompTrainFull, TrainFully)
    #Train1['Predict'] = KNNModel.predict(PrinCompTrain1)
    Test['Predict'] = KNNModel.predict(PrinCompTest)
    print(n_neighbors)
    print(AUC(Testy, Test['Predict']))

#6 again

##### Step3: Final Model using KNN on Full set. Validataion Statistics will not be meaningful #####

TrainFullX = pd.concat([Train1X, Train2X, TestX])
TrainFully = pd.concat([T1y, T2y, Testy], axis= 0)

TrainRecaler = StandardScaler()
TrainRecaler.fit_transform(TrainFullX)
TrainRecaled = pd.DataFrame(Train1Recaler.transform(TrainFullX))

pca = PCA(n_components=45)
pca.fit_transform(TrainRecaled)
PrinCompTrainFull = pd.DataFrame(pca.transform(TrainRecaled))

KNNModel = KNeighborsClassifier(n_neighbors=6, weights = 'distance')
KNNModel.fit(PrinCompTrainFull, TrainFully)

TrainFullX['Predict'] = KNNModel.predict(PrinCompTrainFull)
print(AUC(TrainFully, TrainFullX['Predict'])) #not really meaningful

##### step 4: Final Scoring #####

# because sklearn train test split was used to break the sets apart,
# and I have not been careful with maintaining the index, will repull the data

HO = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/HoldOut.csv', sep = '|')
Training = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/TrainFullProcessed.csv',
                       sep = '|')
Trainingy = Training[['y']]

HO = Tweaks(HO)
Training = Tweaks(Training)

HO = HO[BetterThanRandom]
Training = Training[BetterThanRandom]

HORescaled = pd.DataFrame(Train1Recaler.transform(HO))
TrainingRescaled = pd.DataFrame(Train1Recaler.transform(Training))

PrinCompHO = pd.DataFrame(pca.transform(HORescaled))
PrinCompTraining = pd.DataFrame(pca.transform(TrainingRescaled))

PrinCompHO['PredictKNN'] = KNNModel.predict(PrinCompHO)
PrinCompTraining['PredictKNN'] = KNNModel.predict(PrinCompTraining)

ExportHO = PrinCompHO[['PredictKNN']]
ExportTrain = PrinCompTraining[['PredictKNN']]
ExportTrain.shape
ExportHO.shape
ExportHO.describe()

ExportHO.dtypes

ExportTrain.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/KNNTrain.csv', index = False)
ExportHO.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/KNNHO.csv', index = False)

AUC(Trainingy, PrinCompTraining['PredictKNN'])


