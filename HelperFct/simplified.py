0########################################################################################################################
#
# Remarks:
#   Preliminary variable processing done in ExercisizeLoadData.py
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

Train1 = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/Train1.csv', sep = '|')
Train2 = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/Train2.csv', sep = '|')
Test = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/Test.csv', sep = '|')
HO = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/HoldOut.csv', sep = '|')

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
TestX = Test[BetterThanRandom]
Testy = Test[['y']]
HOX = HO[BetterThanRandom]

Train2X = Train2[BetterThanRandom]
Train2y = Train2[['y']]

if (False):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV

    if (True):

        # no need to rerun

        CVDetails = KFold(n_splits=2, shuffle=True, random_state=12345).split(X=Train1X, y=Train1y)

        param_grid = {
            'num_leaves': [3, 7, 15, 31, 63],
            'feature_fraction': [0.5, .9, 1.0],
            'bagging_fraction': [0.5, 0.8, 1.0],
            'min_data_in_leaf':  [50, 100, 250, 500]
        }

        # ultimately the above represent complexity and how much variability to put into the boosting algorithm

        lgb_estimator = lgb.LGBMClassifier(boosting_type = 'gbdt',
                                           objective = 'binary',
                                           num_boost_round = 1000, learning_rate = 0.05,
                                           metric = 'auc',
                                           verbose = 0)

        #categorical_feature=indexes_of_categories could use later, but I got bored

        gsearch = GridSearchCV(estimator = lgb_estimator,
                               param_grid = param_grid,
                               cv = CVDetails)

        lgb_model = gsearch.fit(X = Train1X, y = Train1y)

        print(lgb_model.best_params_, lgb_model.best_score_) # this is kind of ludicrously good

##### step 2: Tune stopping using Train 2 data #####

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 1.0,
    'bagging_fraction': 0.5,
    'num_leaves': 63, #trying deeper may improve fit: its on the edge of search params
    'learning_rate': 0.05,
    'min_data_in_leaf': 250
}


Train1GBM = lgb.Dataset(Train1X, Train1y)
Train2GBM = lgb.Dataset(Train2X, Train2y)

gbm = lgb.train(params,
                Train1GBM,
                num_boost_round=1000,
                valid_sets=Train2GBM,
                early_stopping_rounds=5)

##### Step 3: Build Final Model T1 + T2 stoppin on Test#####

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 1.0,
    'bagging_fraction': 0.5,
    'num_leaves': 63, #trying deeper may further improve fit: its on the edge of search params
    'learning_rate': 0.05,
    'min_data_in_leaf': 250
}


TrainFull = pd.concat([Train1, Train2], axis=0)
TrainFully = TrainFull['y']


TrainFullX = pd.concat([Train1X, Train2X], axis= 0 )
TrainFully = pd.concat([Train1y, Train2y], axis= 0 )

TrainFullGBM = lgb.Dataset(TrainFullX, TrainFully)
TestGBM = lgb.Dataset(TestX, Testy)

gbm = lgb.train(params,
                TrainFullGBM,
                num_boost_round=1000,
                valid_sets=TestGBM,
                early_stopping_rounds=5)

##### step 4: Final scoring #####

HO = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/HoldOut.csv',
                 sep = '|')
Training = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/TrainFullProcessed.csv',
                       sep = '|')


Training = Tweaks(Training, 'Training')
HO = Tweaks(HO, 'HoldOut')

TrainingX = Training[BetterThanRandom]
Training1y = Training[['y']]

HOX = HO[BetterThanRandom]

Training['PredictGBM'] = gbm.predict(data=TrainingX)
HOX['PredictGBM'] = gbm.predict(data=HOX)

Training[['PredictGBM']].to_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/FinalScores/TrainGBM.csv',
                                index = False)

HOX[['PredictGBM']].to_csv('C:/Users/szreb/Documents/CodeSandBox/Exercisize/Data/FinalScores/HOGBM.csv',
                                index = False)

# quick check that scoring of original data worked
from sklearn.metrics import roc_auc_score as AUC
AUC(Training1y, Training['PredictGBM'])



