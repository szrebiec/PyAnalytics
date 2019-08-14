

import pandas as pd



LogFile = '/ap1/13862_IoT/code/Flo/EDA/Output/01BaseDataAssembly_Log.csv'
pd.DataFrame({'Description': ['Log For 01BaseDataAssembly.py'], 'Details': ['']}).to_csv(LogFile, mode = 'w', header=True, index=False)

def LogIt(description='', value='', file=LogFile):

    if type(value) == str:
        temp = pd.DataFrame({'Description': [description], 'Details': [value]})
        temp = temp[['Description', 'Details']]
        temp.to_csv(file, mode='a', header=False, index=False)
        return 'string line writen to log'
    elif type(value) == pd.core.series.Series:
        temp = value.to_frame().reset_index()
        temp['Description'] = description
        cols = temp.columns.tolist()
        temp = temp[ cols[-1:] + cols[:-1]]
        temp.to_csv(file, mode='a', header=False, index=False)
        return 'Series lines writen to log'
    elif type(value) == pd.core.frame.DataFrame:
        temp = value.reset_index()
        temp['Description'] = description
        cols = temp.columns.tolist()
        temp = temp[ cols[-1:] + cols[:-1]]
        temp.to_csv(file, mode='a', header=True, index=False)
        return 'DF lines writen to log'
    else:
        return 'Unsupported datatype. Nothing output'


#LogIt('Device file shape = ', str(Device.shape))
#LogIt('Device dupe device keys = ', str(Device.DEVICE_ID.duplicated().sum()))
#LogIt('Device proc contents', Device.dtypes)
#LogIt('Device describe', Device.describe())
