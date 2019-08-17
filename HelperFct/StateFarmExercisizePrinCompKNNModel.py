########################################################################################################################
# Code to solve state farms interview- Question 1.-build a K nearest neighbor model:
#
# Remarks:
#   Preliminary variable processing done in StateFarmExercisizeLoadData.py
#   Basic variable selection begun in VariableEDA.ipynb
#
# Step 0: Header & load data
#      1: Variable Processing and PCA
#      2: KNN using 9 neighbors, selected based on AUC erring towards more local knowledge
#      3: Final Scorign
#
########################################################################################################################

##### Step 0: Header #####

import os
import gc
import numpy as np
import pandas as pd

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
    df.drop(columns=['x34', 'x93', 'x35rec', 'x68rec', 'Unnamed: 0'], inplace= True)

    return df

Train1 = Tweaks(Train1)
Train2 = Tweaks(Train2)
Test = Tweaks(Test)


T1y = Train1['y']
T2y = Train2['y']
Testy = Test['y']

Train1.drop(columns = 'y', inplace= True)
Train2.drop(columns = 'y', inplace= True)
Test.drop(columns = 'y', inplace= True)

##### Step1: Variable processing #####

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder as OneHot

Train1Recaler = StandardScaler()
Train1Recaler.fit_transform(Train1)
Train1Recaled = pd.DataFrame(Train1Recaler.transform(Train1))
Train2Recaled = pd.DataFrame(Train1Recaler.transform(Train2))
TestRecaled = pd.DataFrame(Train1Recaler.transform(Test))


pca = PCA(n_components=45)
pca.fit_transform(Train1Recaled)

PrinCompTrain1 = pd.DataFrame(pca.transform(Train1Recaled))
PrinCompTrain2 = pd.DataFrame(pca.transform(Train2Recaled))
PrinCompTest = pd.DataFrame(pca.transform(TestRecaled))

##### Step1: build KNN model #####

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score as AUC

for n_neighbors in range(3, 15):
    KNNModel = KNeighborsClassifier(n_neighbors= n_neighbors)
    KNNModel.fit(PrinCompTrain1, T1y)
    #Train1['Predict'] = KNNModel.predict(PrinCompTrain1)
    Train2['Predict'] = KNNModel.predict(PrinCompTrain2)
    print(n_neighbors)
    print(AUC(T2y, Train2['Predict']))

##### Step2: Final Model: use KNN on Full set. Validataion Statistics will not be meaningful#####

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score as AUC

#Train2.drop(columns='Predict', inplace=True)

TrainFull = pd.concat([Train1, Train2, Test])
TrainFully = pd.concat([T1y, T2y, Testy], axis= 0)

TrainFull.dtypes
TrainFull['My'].describe()

TrainRecaler = StandardScaler()
TrainRecaler.fit_transform(TrainFull)
TrainRecaled = pd.DataFrame(Train1Recaler.transform(TrainFull))
pca = PCA(n_components=45)
pca.fit_transform(TrainRecaled)
PrinCompTrainFull = pd.DataFrame(pca.transform(TrainRecaled))

KNNModel = KNeighborsClassifier(n_neighbors=9)
KNNModel.fit(PrinCompTrainFull, TrainFully)

TrainFull['Predict'] = KNNModel.predict(PrinCompTrainFull)
print(AUC(TrainFully, TrainFull['Predict'])) #not really meaningful

##### step 3: Final Scoring #####
# because sklearn was used to break the sets upart, and I have not been careful with maintaining the index, will repull the data

HO = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/HoldOut.csv', sep = '|')
Training = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/TrainFullProcessed.csv',
                       sep = '|')
Trainingy = Training[['y']]

HO = Tweaks(HO)
Training = Tweaks(Training)

HO['My'] = 0 # missing indicator for y , it is (also 0 on training)
Training.drop(columns = 'y', inplace= True)

HORescaled = pd.DataFrame(Train1Recaler.transform(HO))
TrainingRescaled = pd.DataFrame(Train1Recaler.transform(Training))

PrinCompHO = pd.DataFrame(pca.transform(HORescaled))
PrinCompTraining = pd.DataFrame(pca.transform(TrainingRescaled))

PrinCompHO['Predict'] = KNNModel.predict(PrinCompHO)
PrinCompTraining['Predict'] = KNNModel.predict(PrinCompTraining)

ExportHO = PrinCompHO[['Predict']]
ExportTrain = PrinCompTraining[['Predict']]
ExportTrain.shape

ExportTrain.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/KNNTrain.csv', index = False)
ExportHO.to_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/FinalScores/KNNHO.csv', index = False)

from sklearn.metrics import roc_auc_score as AUC
AUC(Trainingy, PrinCompTraining['Predict'])


