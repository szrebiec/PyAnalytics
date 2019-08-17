########################################################################################################################
# Code to solve state farms interview- via build a logit model BUT THIS WAS NOT FOUND TO BE A FRUITFUL APPROACH:
#   This was my first stab and it was one that was far too manual. A few vars in I ran into a perfect corr 0.998 post
#       Transformation and I decided to abandon this approach in favor of princomp + lasso + logit model. That too
#       was abandoned as I decided that the data is not random in nature, the GBM and the simpler (but weaker) KNN
#       model drove that home. If you want to see how I think about a glm... feel free to keep reading.
#       I am a bit more manual than most, relying on intuirion and manual transforms, which are then supplemented with
#       more advanced techniques.
#
# Remarks:
#   Preliminary variable processing done in StateFarmExercisizeLoadData.py
#   Basic variable selection begun in VariableEDA.ipynb
#
# Step 0: Header & load data
#      1: Manual transforms
#      2: simple model
#      3: Diagnostics
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

Train1['Sample'] = 'Train1'
Train2['Sample'] = 'Train2'

Train1['one'] = 1
Train2['one'] = 1

##### step 1: Manual transforms: if this had gone further these would have been fct'ed up #####

Train1[['x37', 'x75', 'x58', 'x97', 'x41_rec_num', 'x99']].corr()
#'x96' has high multicolinearity

# initial variable transforms
#binding curves
Train1['x37BC10w8'] = np.exp((Train1['x37'] -10 )/ 8)/ (1+ np.exp((Train1['x37'] -10) / 8))
Train1['x96BC20w15'] = np.exp((Train1['x37'] -20 )/ 15)/ (1+ np.exp((Train1['x37'] -20) / 15))

def Bounds(df,xVarname, LB, UB):
    dflocal = df[[xVarname]]
    dflocal.loc[dflocal[xVarname] < LB, xVarname] = LB
    dflocal.loc[dflocal[xVarname] > UB, xVarname] = UB
    return dflocal[xVarname]

# Begin: 1 pct and 99 pct
Train1['x75_1pm'] = Bounds(df = Train1 , xVarname = 'x75',  LB = -12.24628503491328, UB = 13.452484472438035)
Train1['x58_1pm'] = Bounds(df = Train1 , xVarname = 'x58',  LB = -74.1078061710105, UB = 65.14477665240055)
Train1['x41_rec_num_1pm'] = Bounds(df = Train1 , xVarname = 'x41_rec_num',  LB = -2382.7504999999996, UB =2290.0325999999995)
Train1['x83_1pm'] = Bounds(df = Train1 , xVarname = 'x83',  LB = -39.90069874437895, UB = 39.635745842160425)
# end: 1 pct and 99 pct

#custom bounds
Train1['x97_1p5m'] = Bounds(df = Train1 , xVarname = 'x97',  LB = -7.237876750281334, UB = 8.97211638957717)
Train1['x99_UB4LB_9'] = Bounds(df = Train1 , xVarname = 'x99',  LB = -9, UB = 4)
Train1['x99_UB4LB_9'] = Bounds(df = Train1 , xVarname = 'x99',  LB = -9, UB = 4)

from statsmodels.stats.outliers_influence import variance_inflation_factor

UsedVars = ['x37BC10w8', 'x75_1pm', 'x58_1pm','x97_1p5m', 'x41_rec_num_1pm', 'x83_1pm']

##### step2: model ###
FirstStab = sm.Logit(Train1y, sm.add_constant(Train1[UsedVars ]) )
result = FirstStab.fit()
print(result.summary())
print(Train1[UsedVars].corr())

#, 'x99_UB4LB_9' too much multicolinearity
# 96 is perfectly correlated with 37, post transformation I am getting 0.998477  :p
# ok, so the same variables repeatedly in the data and made this approach a wild goose chase, i.e. you
#   REALLY want me to not component up, then do variable selection.
#   This tends to not be the way I build linear models as, I tend to shoot for explainability, and
#   a 1 standard deviation change in PC 1 leading to a 20% increase in log odds is kind of useless.
#   tho I am not sure the value of a 1 std chang in x75, so fair enough.


##### step 3: just for the hallibut some diagnostics ######
Train1['Predict'] = result.predict()
Train1['one'] = 1

def LazyKS(XvarName, df = Train1, WtVarName = 'one', LossVarName = 'y'):
    # Calculates a weighted KS, using Random number to approximately deal with deiscreteness, rather than aggregation
    # bit lazy passing data frames...
    dflocal = df[[XvarName, WtVarName, LossVarName]].sort_values([XvarName]).reset_index()
    dflocal['CDFWt'] = dflocal[WtVarName].cumsum() /dflocal[WtVarName].sum()
    dflocal['CDFy'] = dflocal[LossVarName].cumsum() /dflocal[LossVarName].sum()
    return max(np.abs(dflocal['CDFy'] - dflocal['CDFWt']))

LazyKS(XvarName='Predict', df = Train1, WtVarName = 'one', LossVarName = 'y')