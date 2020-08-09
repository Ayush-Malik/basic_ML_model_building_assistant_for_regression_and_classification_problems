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
            Strategy[i] = 'Drop it'
    missing_values_count['Data_Type']                 = Data_Types 
    missing_values_count['Strategy_that_can_be_used'] = Strategy

    return(missing_values_count)

def heatmap_generator(data, yticklabel=False, cbar_value=False):
    '''
    Generates heatmap plot according to the data received;
    yticklabel:- defaults False but also can take values True to display all y labels and "auto" to display values automatically.
    cbar_value:- default is False but can also takes True as input to show cbar of heatmap.
    
    Example
    =========
    >>>heatmap_generator(df.isnull())
    >>>plots heatmap to display all null values in the dataset.
    '''
    return(sns.heatmap(data, yticklabels = yticklabel, cbar = cbar_value))

def imbalanced_feature(df):
    dic = dict(df.dtypes)

    categorical_features = []

    for val in dic:
        if dic[val] == 'object':
            categorical_features.append(val)
            

    print(categorical_features)


    imbalanced_features = []


    for feature_name in categorical_features: # Checking only categorical features , if they are balanced or imbalanced
        cool = df[feature_name].value_counts()
        
        if len( cool.value_counts().index ) <= int(0.05 * len(df)): # Comparing number of categories in a feature with number of rows , this will help us to reduce unnecessary usage of computational power     
            new_dic = dict(  ( df[feature_name].value_counts() / len(df) ) * 100  )
            for val in new_dic: # Checking percentage of each categories of a feature , if percentage of a category of a particular feature is above 80 percent , then mark that feature as imbalanced feature
                if new_dic[val] >= 90:
                    imbalanced_features.append( feature_name )
    return(imbalanced_features)
def cat_num(df):
    dic = dict(df.dtypes)

    categorical_features = []
    numerical_features = []

    for val in dic:
        if dic[val] == 'object':
            categorical_features.append(val)
        else:
            numerical_features.append(val)
    return(categorical_features)

def prcntage_values( categorical_feature, df):
    feature_df = pd.DataFrame( dict((  df[categorical_feature].value_counts() )).items() , columns = ['Category' , '%age'] )
    return px.pie( feature_df , values='%age', names='Category', title='Category vs %age for ' + categorical_feature + ' ' ,color_discrete_sequence=px.colors.sequential.RdBu)

# def pie_plotter( lis ):
#     amazing = pd.DataFrame( lis , columns = ['Combined_Category_Type' , '%age'])
# #     amazing.iplot(kind = 'bar' , x = 'Combined_Category_Type' , y = '%age' , title = 'Category vs percentage plot' , xTitle = 'Category' , yTitle = '%age')

#     return px.pie( amazing , values='%age', names='Combined_Category_Type', title='Category vs %age' ,color_discrete_sequence=px.colors.sequential.RdBu)

# def different_cat_comparator( lis_of_feat, df): # here function accepts lis size == 2 only 
#     type_1 , type_2  = lis_of_feat[0] , lis_of_feat[1]
#     lis = []
#     for cat_1 in df[type_1].value_counts().index: 
#         sub_lis = []
#         for cat_2 in df[type_2].value_counts().index:

#             new = df[ ( df[type_1] == cat_1 ) & (df[type_2] == cat_2) ]
#             sub_lis.append( [ str(cat_1) + ' and ' + str(cat_2) ,   len(new)  ] )
#         lis.append(sub_lis)
#     for val in lis :
#         p = pie_plotter(val)
#         return(p)
    

# ---------------------------------------------------------------------------
# def pie_plotter( lis ):
#     amazing = pd.DataFrame( lis , columns = ['Combined_Category_Type' , '%age'])
#     return px.pie( amazing , values='%age', names='Combined_Category_Type', title='Category vs %age' ,color_discrete_sequence=px.colors.sequential.RdBu)

# def different_cat_comparator( lis_of_feat, df): # here function accepts lis size == 2 only 
#     type_1 , type_2  = lis_of_feat[0] , lis_of_feat[1]
#     lis = []
#     for cat_1 in df[type_1].value_counts().index: 
#         sub_lis = []
#         for cat_2 in df[type_2].value_counts().index:

#             new = df[ ( df[type_1] == cat_1 ) & (df[type_2] == cat_2) ]
#             sub_lis.append( [ str(cat_1) + ' and ' + str(cat_2) ,   len(new)  ] )
#         lis.append(sub_lis)
        
#     return [ pie_plotter(val)  for val in lis]
# ---------------------------------------------------------------------------


def two_cat_comparator( lis_of_feat , df ):
    type_1 , type_2  = lis_of_feat[0] , lis_of_feat[1]
    dic = {}
    for cat_1 in df[type_1].value_counts().index: 
        sub_lis = []
        for cat_2 in df[type_2].value_counts().index:

            new = df[ ( df[type_1] == cat_1 ) & (df[type_2] == cat_2) ]
            sub_lis.append( [ cat_2 ,   len(new)  ] )
        dic[cat_1] = sub_lis
   
 
    wow = []

    for key in dic:
        wow.append(  go.Bar(name = key  , x = np.array(dic[key])[: , 0]  , y = np.array(dic[key])[: , 1] )  )
    fig = go.Figure(data = wow)
    fig.update_layout(barmode='group')
    
    return fig