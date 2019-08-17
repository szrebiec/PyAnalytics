########################################################################################################################
# Code to solve state farms interview- by building a GBM model:
#
# Remarks:
#   Preliminary variable processing done in StateFarmExercisizeLoadData.py
#   Basic variable selection begun in VariableEDA.ipynb. The key ingredient was identifying some vars that I won't be
#       using
#
# Step 0: Header & load data
#      1: Light GBM Grid search
#      2: Finish Train 1 model
#      3: Build Train 1 and Train 2 modely
#      4: Scoring... it would be better to break this off, and save the parameters, but I got lazy
#
#  This code was built in phases, and a bit sloppily all phases are thrown together.
#       phase 1: Build on Train1 using 2 fold CV
#       phase 2: Build on Train1 use Train2 as testing and stopping parameters sample.
#       phase 3: Build on Train1 &Trains 2 stop based on testing value
#
########################################################################################################################

##### Step 0: Header #####

import os
import gc
import numpy as np
import pandas as pd

import lightgbm as lgb

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

Train1 = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Train1.csv', sep = '|')
Train2 = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Train2.csv', sep = '|')
Test = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Test.csv', sep = '|')
HO = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/HoldOut.csv', sep = '|')

def Tweaks(df, Sample):
    # minor data changes
    df['Sample'] = Sample
    df['one'] = 1

    df.rename(columns={'Unnamed: 0': 'OriginalSeqNumber'}, inplace=True)

    # collapsing thin days with adjacent days
    df['dayInd1'] = ((df['x35rec'] == 'monday' ) | (Train1['x35rec'] == 'tuesday')) + 0
    df['dayInd2'] = ((df['x35rec'] == 'thurday' ) | (Train1['x35rec'] == 'tuesday') | (Train1['x35rec'] == 'friday'))+ 0
    df['dayInd3'] = (df['x35rec'] == 'wednesday' )  + 0

    # I enjoyed that Spelling 'mistake' ;) I would have gone with Thorday tho

    return df

Train1 = Tweaks(Train1, 'Train1')
Train2 = Tweaks(Train2, 'Train2')
Test = Tweaks(Test, 'Test')
HO = Tweaks(HO, 'HoldOut')

# The following list was used at one time.
BetterThanRandom = ['x37', 'x58', 'x75', 'x97', 'x41_rec_num', 'x70', 'x51', 'x2', 'x66', 'x96', 'x56',
                    'x1', 'x78', 'x79', 'x99', 'x40', 'x73', 'x72', 'x63', 'x5', 'x22', 'x83', 'x69', 'x33',
                    'x3', 'x45_rec_num', 'x50', 'x85', 'x44', 'x21', 'x20', 'x10', 'x0', 'x68SummerIndicator',
                    'x68FallIndicator', 'x38', 'x8', 'x74', 'x92', 'x53', 'x48', 'x29', 'x67', 'x95', 'x80',
                    #'x68SpringIndicator', delibrately left out to implicitly combine with the tiny winter sample, maybe combine with Fall?
                    'x52','dayInd1', 'dayInd2', 'dayInd3'
                    ]

# based on KS, the other vars are about as predictive as a random numbers

##### Step1: Light GBM Grid search #####

Train1X = Train1[BetterThanRandom]
Train1y = Train1[['y']]

Train2X = Train2[BetterThanRandom]
Train2y = Train2[['y']]

TestX = Test[BetterThanRandom]
Testy = Test[['y']]
HOX = HO[BetterThanRandom]

Train2X = Train2[BetterThanRandom]
Train2y = Train2[['y']]

from sklearn.model_selection import GridSearchCV

# no need to rerun

CVDetails = KFold(n_splits=2, shuffle=True, random_state=12345).split(X=Train1X, y=Train1y)

param_grid = {
    'num_leaves': [3, 7, 15, 31, 63],
    'feature_fraction': [0.5, .9, 1.0],
    'bagging_fraction': [0.5, 0.8, 1.0],
    'min_data_in_leaf':  [50, 100, 250, 500]
}

# ultimately the above represent complexity and how much variability to put into the boosting algorithm

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

param_grid = {
    'n_estimators': [6, 10, 35],
    'max_features': [0.5, 0.75, 1.0],
    'min_samples_split':  [25, 50, 100, 250, 500]
}

RFModel = RF(bootstrap = True,
             random_state = 31415,
             verbose = True)

gsearch = GridSearchCV(estimator = RFModel,
                       param_grid = param_grid)

RFModelFitted = gsearch.fit(X=Train1X, y=Train1y)

RFModel = RF(n_estimators = 10,
             min_samples_split = 100,
             max_features = 0.9,
             bootstrap = True,
             random_state = 31415,
             verbose = False)

RFModelFitted = RFModel.fit(X = Train1X, y = Train1y)
Train2['Pred'] = RFModelFitted.predict(Train2X)

print(RFModelFitted.best_params_, RFModelFitted.best_score_)

#Horrible overfit
AUC(Train1y, Train1['Pred']) #
AUC(Train2y, Train2['Pred']) #


##### A more manual approach #####

for n_estimators in [6, 10, 20, 35]:
    RFModel = RF(n_estimators = n_estimators,
                 max_features=0.9,
                 bootstrap=True,
                 random_state=31415,
                 min_samples_split = 500)

    RFModelFitted = RFModel.fit(X=Train1X, y=Train1y)
    Train2['Pred'] = RFModelFitted.predict(Train2X)
    print(n_estimators)
    print(AUC(Train2y, Train2['Pred']))

#wee: n_estimators in [6, 10, 20, 35]
#max_features = 0.9,
#bootstrap = True,
#random_state = 31415,
#min_samples_split = 500 all bad

for n_estimators in [6, 10, 20, 35]:
    RFModel = RF(n_estimators=n_estimators,
                 max_features=0.9,
                 bootstrap=False,
                 random_state=31415,
                 min_samples_split=500)

    RFModelFitted = RFModel.fit(X=Train1X, y=Train1y)
    Train2['Pred'] = RFModelFitted.predict(Train2X)
    print(n_estimators)
    print(AUC(Train2y, Train2['Pred']))

# BS = False is better?

for x in [0.5, 0.9, 1.0]:
    RFModel = RF(n_estimators=25,
                 max_features=x,
                 bootstrap=False,
                 random_state=31415,
                 min_samples_split=500)

    RFModelFitted = RFModel.fit(X=Train1X, y=Train1y)
    Train2['Pred'] = RFModelFitted.predict(Train2X)
    print(x)
    print(AUC(Train2y, Train2['Pred']))
# higher is better.


q