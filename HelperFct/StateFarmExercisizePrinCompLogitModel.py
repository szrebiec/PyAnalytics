########################################################################################################################
# Code to solve state farms interview- by build a logit model BUT THIS WAS NOT FOUND TO BE A FRUITFUL APPROACH
#   Princomping proved to be a good idea. Unlike real random data, EVERYTHING was useful and because of that
#   This approach was abandonded for GBM which is more flexible, and the not quite as well performing k-nearest neigh.
#
# Remarks:
#   Preliminary variable processing done in StateFarmExercisizeLoadData.py
#   Basic variable selection begun in VariableEDA.ipynb
#
# Step 0: Header & load data
#      1: PCA
#      2: Lasso
#      3: Logit model, was worried about a poor set up on lasso.
#      4: Diagnostics
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

pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/Test.csv', sep = '|').shape

Train1['Sample'] = 'Train1'
Train2['Sample'] = 'Train2'

Train1['one'] = 1
Train2['one'] = 1

TopPredList = ['x37', 'x75', 'x58', 'x97', 'x41_rec_num', 'x99', 'x96',
                    'x83', 'x51', 'x56', 'x79', 'x70', 'x66', 'x72', 'x1',
                    'x33', 'x63', 'x22', 'x5', 'x2', 'x78', 'x50', 'x40',
                    'x69', 'x3', 'x21', 'x73', 'x85', 'x20', 'x45_rec_num',
                    'x44', 'x10', 'x0','x68SummerIndicator']

##### Step 1: compute principle components #####
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Train1Recale =  StandardScaler().fit_transform(Train1[TopPredList])
pca = PCA(n_components=25)
PrinComp = pd.DataFrame(pca.fit_transform(Train1Recale))
PrinComp.head()

colnames = []
for Index in PrinComp.columns:
    colnames.append('PC' + str(Index))

PrinComp.columns = colnames

##### Step 2: Lasso Model on top 35'ish predictors #####

from sklearn.linear_model import LogisticRegression

Lasso = LogisticRegression(penalty='l1', verbose = 1)
Lasso.fit(PrinComp, Train1['y'])

Lasso.verbose
Lasso.coef_

##### step 3: Dig a bit deeper with logit model #####

VarList = ['PC0', #'PC1', 'PC2',
           'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
           'PC11', 'PC12', 'PC13', 'PC14',
           #'PC15',
           'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
           'PC21', 'PC22']

FirstStab = sm.Logit(Train1['y'], sm.add_constant(PrinComp[VarList] ) )
result = FirstStab.fit()
print(result.summary())

PrinComp['Predict'] = result.predict()
PrinComp['one'] = 1
PrinComp['y'] = Train1['y']

##### step 4: diagnostics #####

def LazyKS(XvarName, df = Train1, WtVarName = 'one', LossVarName = 'y'):
    # Calculates a weighted KS, using Random number to approximately deal with deiscreteness, rather than aggregation
    # bit lazy passing data frames...
    dflocal = df[[XvarName, WtVarName, LossVarName]].sort_values([XvarName]).reset_index()
    dflocal['CDFWt'] = dflocal[WtVarName].cumsum() /dflocal[WtVarName].sum()
    dflocal['CDFy'] = dflocal[LossVarName].cumsum() /dflocal[LossVarName].sum()
    return max(np.abs(dflocal['CDFy'] - dflocal['CDFWt']))

LazyKS(XvarName='Predict', df = PrinComp, WtVarName = 'one', LossVarName = 'y')

# results are shockingly good