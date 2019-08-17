
from

import sklearn2pmml as skpmml
import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn2pmml import PMMLPipeline, sklearn2pmml
import lightgbm as lgb
import pandas as pd
import numpy as np

Train1X = pd.read_csv('C:\\Users\\szreb\\Downloads\\lightgbmDataX.csv', sep = '|')
Train1X.drop(columns=['Unnamed: 0'], inplace = True)

Train1y = pd.read_csv('C:\\Users\\szreb\\Downloads\\lightgbmDatay.csv', sep = '|')
Train1y.drop(columns=['Unnamed: 0'], inplace = True)

pipe = PMMLPipeline([
    ('lgb', lgb.LGBMRegressor(min_data=1))  # min_data is needed because of small dataset
])


pipe.fit(Train1X, Train1y)
sklearn2pmml(pipe, 'C:\\Users\\szreb\\Downloads\\lightgbmDataX.csv\\x.pmml')


QC = pd.read_csv('C:\\Users\\szreb\\Downloads\\lightgbm.txt')

LGBModel = lgb.Booster(model_file='C:\\Users\\szreb\\Downloads\\lightgbm.txt')



pipe = PMMLPipeline([
    ('lgb', LGBModel)
])



sklearn2pmml(pipe, './x.pmml')

#######

pipe = PMMLPipeline([
    ('lgb', lgb.LGBMRegressor(min_data=1))  # min_data is needed because of small dataset
])

X = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3], 'B': np.random.normal(size=6)})
y = pd.Series(np.random.normal(size=6))

pipe.fit(X, y)
sklearn2pmml(pipe, './x.pmml')  # This one runs ok

X['A'] = X['A'].astype('category')

pipe.fit(X, y)
sklearn2pmml(pipe, './x.pmml')  # This one fails

#####






skpmml.


QC.head()
