import pandas as pd
import numpy as np
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

import cufflinks as cf
cf.go_offline()

def type_of_feature(df):

    dic = dict(df.dtypes)
    dic = pd.DataFrame(dic.items(), columns = ["Features", "Dtypes"]).set_index("Features")
    return(dic)

def null_value(df):
    # Null values management system (^_^)

    missing_values_count = df.isnull().sum().sort_values(ascending = False)
    missing_values_count = missing_values_count.head( df.shape[1] - list(missing_values_count).count(0) )
    missing_values_count = missing_values_count.to_frame().reset_index().rename( columns = {'index' : 'Column/Feature' , 0 : '%age_Null_val_count'})

    missing_values_count['%age_Null_val_count'] = ( missing_values_count['%age_Null_val_count'] / len(df) ) * 100

    Data_Types = []
    Strategy   = []

    for feature in missing_values_count['Column/Feature']:
        Data_Types.append( df.dtypes[feature]  )
        if df.dtypes[feature] == 'object':
            Strategy.append('mode')
        else:
            Strategy.append('mean')

    for i in range(len(missing_values_count)):
        if missing_values_count['%age_Null_val_count'][i] >= 85 :
            Strategy[i] = ['Drop it']
    missing_values_count['Data_Type']                 = Data_Types 
    missing_values_count['Strategy_that_can_be_used'] = Strategy

    return(missing_values_count)