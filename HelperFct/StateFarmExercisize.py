########################################################################################################################
# Code to solve state farms interview- Question 1.-Clean and prepare your data:
#
#"There are several entries where values have been deleted to simulate dirty data. Please clean the data with whatever
# method(s) you believe is best/most suitable. Note that some of the missing values are truly blank (unknown answers).
# Success in this exercise typically involves feature engineering and avoiding data leakage.
#
# Step 0: Header
#      1: Helper fcts
#      2: # x93
# Remarks: missingness is not realistic-missing rates are low- adding all missing flags has a max of 1.
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


##### Step1: helper fcts #####

TrainFull = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/exercise_05_train.csv',
                        sep = ',')

# logic fcts

def DateStandardization(value):
    if value == 'wed':
        return 'wednesday'
    elif value == 'thur':
        return 'thursday'
    elif value == 'fri':
        return 'friday'
    else:
        return value

def MonthStandardization(value):
    # Sample is highly unbalanced/no winter observations
    if value == 'July':
        return 'Jul'
    elif value == 'sept.':
        return 'Sep'
    elif value == 'January':
        return 'Jan'
    elif value == 'Dev':
        return 'Dec'
    else:
        return value

# Process unusual data

Xvars = TrainFull.columns.drop('y')

def ProcessDataFrame(df, testing = False):
    DropList = []
    Mxvars = [] #missing indicators

    # awkward way of writing currency treated as numeric
    df['x41_rec_num'] = df.x41.str.replace('$', '').astype(float)
    DropList.append('x41')

    # awkward way of writing pct treated as numeric
    df['x45_rec_num'] = df.x45.str.replace('%', '').astype(float)
    DropList.append('x45')

    #clean some spellings
    df['x35rec'] = df['x35'].apply(DateStandardization)
    DropList.append('x35')

    df['x68rec'] = df['x68'].apply(MonthStandardization)
    DropList.append('x68')

    #data is badly unbiased wrt season- monitor this

    df['Mx68rec'] = df['x68rec'].isna() + 0
    Mxvars.append('Mx68rec')
    df.loc[df['Mx68rec'] == 1, 'x68rec'] = 'Jul'

    df['x68WinterIndicator'] = ((df['x68rec'] == 'Dec') | (df['x68rec'] == 'Feb') | (df['x68rec'] == 'Jan')) + 0
    df['x68SpringIndicator'] = ((df['x68rec'] == 'Mar') | (df['x68rec'] == 'Apr') | (df['x68rec'] == 'May')) + 0
    df['x68SummerIndicator'] = ((df['x68rec'] == 'Jun') | (df['x68rec'] == 'Jul') | (df['x68rec'] == 'Aug')) + 0
    df['x68FallIndicator'] = ((df['x68rec'] == 'Sep') | (df['x68rec'] == 'Oct') | (df['x68rec'] == 'Nov')) + 0

    #psuedo -one hot

    df['x34Volks'] = (df['x34'] == 'volkswagon') + 0
    df['x34Toyota'] = (df['x34'] == 'Toyota') + 0
    df['x34ChevyChrystler'] = ((df['x34'] == 'chrystler') | (df['x34'] == 'chevrolet') )+ 0
    df['x34BMWMercedes'] = ((df['x34'] == 'bmw') | (df['x34'] == 'mercades')) + 0
    df['x34tesla'] = (df['x34'] == 'tesla') + 0
    df['x34HondaNissanFor'] = ((df['x34'] == 'Honda') | (df['x34'] == 'nissan') | (df['x34'] == 'ford')) + 0
    #DropList.append('x34')

    df['Mx34'] = df['x34'].isna() + 0
    Mxvars.append('Mx34')
    df.loc[df['Mx34'] == 1, 'x34'] = 'volkswagon'


    #revisit post logit?

    #miss indicators

    numeric = ['float64', 'int64']

    NumericVars = list(df.select_dtypes(include=numeric))

    for xvar in NumericVars:
        df['M' + xvar] = df[xvar].isnull() + 0
        Mxvars.append('M' + xvar)

    df['MissCount'] = 0
    for mxvar in Mxvars:
        df['MissCount'] = df[mxvar] + df['MissCount']

    #impute to mean , cause I am lazy and nothing jumps out as being neccessary (fill rates were high):
    # mode for charecteristic data
    df['x0'] = df['x0'].fillna(2.0202547207167654)
    df['x1'] = df['x1'].fillna(-3.924559082841259)
    df['x2'] = df['x2'].fillna(1.006619329692027)
    df['x3'] = df['x3'].fillna(-1.378329627753492)
    df['x4'] = df['x4'].fillna(0.07019867953082548)
    df['x5'] = df['x5'].fillna(-0.7152134424059862)
    df['x6'] = df['x6'].fillna(-0.0027058611284104675)
    df['x7'] = df['x7'].fillna(-0.02568887935298335)
    df['x8'] = df['x8'].fillna(-0.35480791715146437)
    df['x9'] = df['x9'].fillna(-0.01702389008258654)
    df['x10'] = df['x10'].fillna(6.665975324666405)
    df['x11'] = df['x11'].fillna(0.03492323604171367)
    df['x12'] = df['x12'].fillna(-5.970745301844286)
    df['x13'] = df['x13'].fillna(0.0007680288341640738)
    df['x14'] = df['x14'].fillna(5.928901795571674e-05)
    df['x15'] = df['x15'].fillna(0.0042137789307495075)
    df['x16'] = df['x16'].fillna(-0.022206019773063933)
    df['x17'] = df['x17'].fillna(0.0011413856915879752)
    df['x18'] = df['x18'].fillna(9.541344484278127)
    df['x19'] = df['x19'].fillna(-0.0020052243087945298)
    df['x20'] = df['x20'].fillna(6.004878615780199)
    df['x21'] = df['x21'].fillna(1.1392868873657649)
    df['x22'] = df['x22'].fillna(-1.4259956724863492)
    df['x23'] = df['x23'].fillna(-0.003322032250207968)
    df['x24'] = df['x24'].fillna(0.045901588082808184)
    df['x25'] = df['x25'].fillna(0.009791094492358397)
    df['x26'] = df['x26'].fillna(0.003568470811582933)
    df['x27'] = df['x27'].fillna(3.6606296227687443)
    df['x28'] = df['x28'].fillna(-0.00498574320433296)
    df['x29'] = df['x29'].fillna(0.026591867302784556)
    df['x30'] = df['x30'].fillna(-0.03378577291811226)
    df['x31'] = df['x31'].fillna(0.02163403438140905)
    df['x32'] = df['x32'].fillna(0.0187483799626336)
    df['x33'] = df['x33'].fillna(-1.0088056240713912)
    df['x36'] = df['x36'].fillna(0.001262009708424307)
    df['x37'] = df['x37'].fillna(0.5015929209586155)
    df['x38'] = df['x38'].fillna(0.007088759346479149)
    df['x39'] = df['x39'].fillna(0.010948392127812321)
    df['x40'] = df['x40'].fillna(1.1290554237234331)
    df['x42'] = df['x42'].fillna(-0.6159979697360922)
    df['x43'] = df['x43'].fillna(0.13211817289235323)
    df['x44'] = df['x44'].fillna(-18.847070949935222)
    df['x46'] = df['x46'].fillna(0.01637103556301365)
    df['x47'] = df['x47'].fillna(-0.006229536891114977)
    df['x48'] = df['x48'].fillna(0.07734489529222838)
    df['x49'] = df['x49'].fillna(0.046994009512475045)
    df['x50'] = df['x50'].fillna(8.071981286087626)
    df['x51'] = df['x51'].fillna(-6.839185026511665)
    df['x52'] = df['x52'].fillna(-0.0018409237531313)
    df['x53'] = df['x53'].fillna(0.013845183174917652)
    df['x54'] = df['x54'].fillna(-0.029328824951084752)
    df['x55'] = df['x55'].fillna(0.054399924779595585)
    df['x56'] = df['x56'].fillna(-2.0129460911866723)
    df['x57'] = df['x57'].fillna(-0.004411294979660676)
    df['x58'] = df['x58'].fillna(-4.254642675776443)
    df['x59'] = df['x59'].fillna(-0.03515911990615562)
    df['x60'] = df['x60'].fillna(-0.006142821103793886)
    df['x61'] = df['x61'].fillna(-0.009653647793779174)
    df['x62'] = df['x62'].fillna(-0.004942447211337997)
    df['x63'] = df['x63'].fillna(-2.5191882287599543)
    df['x64'] = df['x64'].fillna(-0.028990296811510448)
    df['x65'] = df['x65'].fillna(-0.01710184603989499)
    df['x66'] = df['x66'].fillna(-1.11621204021584)
    df['x67'] = df['x67'].fillna(-0.07929211935249014)
    df['x69'] = df['x69'].fillna(-1.6621388007364273)
    df['x70'] = df['x70'].fillna(0.5283151120907079)
    df['x71'] = df['x71'].fillna(0.23822629568230033)
    df['x72'] = df['x72'].fillna(-3.0113928369118463)
    df['x73'] = df['x73'].fillna(-5.70646874780868)
    df['x74'] = df['x74'].fillna(0.3553793836511759)
    df['x75'] = df['x75'].fillna(0.8235195114898106)
    df['x76'] = df['x76'].fillna(0.0061543561614243285)
    df['x77'] = df['x77'].fillna(-0.08408642494534697)
    df['x78'] = df['x78'].fillna(-0.9961169798403228)
    df['x79'] = df['x79'].fillna(1.2741211040073723)
    df['x80'] = df['x80'].fillna(-0.10517491134266595)
    df['x81'] = df['x81'].fillna(0.0048285632706532595)
    df['x82'] = df['x82'].fillna(-0.0015694405092496477)
    df['x83'] = df['x83'].fillna(0.7599177354307776)
    df['x84'] = df['x84'].fillna(-0.03857127914056287)
    df['x85'] = df['x85'].fillna(-0.5653479965707627)
    df['x86'] = df['x86'].fillna(0.043029808183689316)
    df['x87'] = df['x87'].fillna(0.0351573331710083)
    df['x88'] = df['x88'].fillna(0.01406346575847559)
    df['x89'] = df['x89'].fillna(0.003357011517530126)
    df['x90'] = df['x90'].fillna(-11.953656452652435)
    df['x91'] = df['x91'].fillna(0.002118152218809467)
    df['x92'] = df['x92'].fillna(0.024485575412597837)
    df['x94'] = df['x94'].fillna(-0.012012250539900819)
    df['x95'] = df['x95'].fillna(0.01912327433385917)
    df['x96'] = df['x96'].fillna(-0.31734498084429474)
    df['x97'] = df['x97'].fillna(-0.5624528442343203)
    df['x98'] = df['x98'].fillna(0.00048433891441449395)
    df['x99'] = df['x99'].fillna(0.17971509003301692)

    df['x41_rec_num'] = df['x41_rec_num'].fillna(-4.383351)
    df['x45_rec_num'] = df['x45_rec_num'].fillna( -0.000012)

    if testing == False:
        df.drop(columns = DropList, inplace=True)

    return df

def LoadTraining():
    TrainFull = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/exercise_05_train.csv',
                            sep=',')
    TrainFull = ProcessDataFrame(TrainFull)
    return TrainFull

def LoadTest():
    TestFull = pd.read_csv('C:/Users/szreb/Documents/CodeSandBox/StateFarmExercisize/Data/exercise_05_test.csv',
                            sep=',')
    TestFull = ProcessDataFrame(TestFull)
    return TestFull

#def LoadTrain1(TrainFull):
#    return train_test_split(TrainFull, test_size=0.5, random_state= 31415)

##### Step 2: miscelaneous eda. #####

##### Debugging zone ####
if __name__ == '__main__':
    TrainFull = LoadTraining()
    Train1, Train2 = train_test_split(TrainFull, test_size=0.5, random_state= 31415)

##### Scratch work #####
#for xvar in xvars:
#    print("df['" + xvar + "'] = df['" + xvar + "'].fillna(", np.mean(TrainFull[xvar]), ')')


