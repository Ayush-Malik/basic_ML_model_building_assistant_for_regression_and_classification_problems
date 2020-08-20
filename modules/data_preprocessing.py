import pandas as pd
import numpy as np

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

import cufflinks as cf
cf.go_offline()

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def type_of_feature(df):

    new_df = dict(df.dtypes)
    new_df = pd.DataFrame(new_df.items(), columns = ["Features", "Dtypes"]).set_index("Features")
    return(new_df)

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
        if missing_values_count['%age_Null_val_count'][i] >= 50 :
            Strategy[i] = 'Drop it'
    missing_values_count['Data_Type']                 = Data_Types 
    missing_values_count['Strategy_that_can_be_used'] = Strategy

    return(missing_values_count)

def heatmap_generator(df , coloraxis_val = False ):
    '''
    Generates heatmap plot according to the data received;
    cbar_value:- default is False but can also takes True as input to show cbar of heatmap.
    
    Example
    =========
    >>> heatmap_generator(df.isnull())
    >>> plots heatmap to display all null values in the dataset.
    '''
    if sum(df.isnull().sum()) != 0:
        fig = px.imshow(df.isnull() , color_continuous_scale = 'ice' , width = 800, height = 600,)
        fig.layout.coloraxis.showscale = coloraxis_val
        fig.layout
        return fig
    else:
        return None

def imbalanced_feature(df):
    dic = dict(df.dtypes)

    categorical_features = []

    for val in dic:
        if dic[val] == 'object':
            categorical_features.append(val)
            

    # print(categorical_features)


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


def two_cat_comparator( lis_of_feat , df ):
    type_1 , type_2  = lis_of_feat[0] , lis_of_feat[1]
    dic = {}
    unique_len_1 = len(df[type_1].value_counts())
    unique_len_2 = len(df[type_2].value_counts())
    
    if unique_len_1 > 20 or unique_len_2 > 20:
        for cat_1 in df[type_1].value_counts().index:
            sub_dict = {}
            for cat_2 in df[type_2].value_counts().index:

                new = df[(df[type_1] == cat_1) & (df[type_2] == cat_2)]
                sub_dict[cat_2] = len(new)
            dic[cat_1] = sub_dict
        return dic
    else:
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




        fig.update_layout(barmode='group' ,
                        xaxis=dict(
                                        title = lis_of_feat[1],
                                        titlefont_size=16,
                                        tickfont_size=14,
                                    ) , 
                        yaxis=dict(
                                        title = "Count",
                                        titlefont_size=16,
                                        tickfont_size=14,
                                    ) , 
                        legend=dict(
                                    title = dict( text = lis_of_feat[0] ,
                                                  font_family = 'Arial' ,
                                                  font_size = 25 )
                                    
                                )
        )
        
        return fig

def missing_value_lis(df):
    missing_values_count = null_value(df)
    return (list(missing_values_count['Column/Feature']))


def drop_feat(df, lis_drop):
    # Time to drop or fill the Nan values in given features
    missing_values_count = null_value(df)
    feature_tracker = list(missing_values_count['Column/Feature'])

    drop_features = lis_drop
    df.drop( drop_features , axis = 1 , inplace = True)

    for feat in drop_features:
        feature_tracker.remove(feat)

    if len(lis_drop) == 1:
        return(feature_tracker, "Feature was Dropped Successfully")
    else:
        return(feature_tracker, "Features were Dropped Successfully")


def fill_feature(df, feature_ch, liss_fill): 
    i = 0   
    for feature_name in feature_ch :
        strategy_given_by_user = liss_fill[i]
        if strategy_given_by_user == 'mean':
            df[ feature_name ] = df[ feature_name ].fillna( df[ feature_name ].mean())
        elif strategy_given_by_user == 'mode':
            df[ feature_name ] = df[ feature_name ].fillna( df[ feature_name ].mode()[0])
        elif strategy_given_by_user == 'median':
            df[ feature_name ] = df[ feature_name ].fillna( df[ feature_name ].median())
        i += 1
    return pd.DataFrame(df.isnull().sum().sort_values(ascending = False)).reset_index().rename(columns = {'index' : 'Feature' , 0 : 'Null Value Count'})


def useless_feat(df):
    useless_ls = []
    for col in df.columns: 
        if df.dtypes[col] == "O" and df[col].nunique() >= 0.05*df.shape[0]:
            useless_ls.append(col)
    useless_df = pd.DataFrame(useless_ls, columns = ["Feature"]) 
    return(useless_df)

def drop_useless_feat(df, feature):
    df.drop([feature], axis = 1, inplace = True)



# subplot makes for table + piechart which will be used in value counter

def suplots_maker_for_table_and_piechart(df , type_null , feature = None):
#-------------------------------------------------------------------------------------------------------
    # Here we are using basically 4 variables , 2 for table and 2 for pie chart
    # 1 Table variables --> headers , value_lis_for_table
                        # headers is basically 1d list --> example --> ['Category' , 'Count']
                        # value_lis_for_table is like given example below :
                                            # [['S' , 'C' , 'Q'] , [644 , 168 , 77]]
    # 2 Pie Chart --> labels , values
                        # labels is basically a 1d list --> example --> ['S' , 'C' , 'Q']
                        # vlaues is basically a 1d list --> example --> [644 , 168 , 77]
#-------------------------------------------------------------------------------------------------------


    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    if type_null == True:
        dic = dict(df[feature].value_counts()) 

        
        # Table parameters --> headers , value_lis_for_table
        headers = ['Category' , 'Count']
        value_lis_for_table = [list( dic.keys() ) ,list( dic.values() ) ]

        # Pie Chart parameters --> labels and values
        labels = value_lis_for_table[0]
        values = value_lis_for_table[1]

        

    else:
        # Table parameters --> headers , value_lis_for_table
        headers = ['Feature', 'Dtype']
        type_of_feat_df = type_of_feature(df)

        dic = {}
        for val in type_of_feat_df.groupby('Dtypes'):
            dic[ str(val[0]) ] =  len(val[1]) 
        
        new_df = type_of_feature(df).reset_index()

        
        A = []
        B = []
        for val in new_df.values:
            A.append(str(val[0]).upper())
            B.append(str(val[1]))

        value_lis_for_table = [A , B]

        # Pie Chart parameters --> labels and values
        labels = list( dic.keys() )
        values = list( dic.values() )


    # trace for table , here extra parameters are just to provide style
    trace_table = go.Table(
                            columnorder = [2,5],
                            columnwidth = [1 , 1],
                            header=dict(
                                values = headers,
                                line_color='darkslategray',
                                fill_color=headerColor,
                                align=['left','center'],
                                font=dict(color='white', size=15)
                            ),
                            cells=dict(
                                values= value_lis_for_table,
                                line_color='darkslategray',
                                # 2-D list of colors for alternating rows
                                # fill_color = [[rowOddColor , rowEvenColor , rowOddColor]*50],
                                fill_color = [[  rowEvenColor*(i%2 != 0) + rowOddColor*(i%2 == 0) for i in range(len( value_lis_for_table[0] ))  ]] , 
                                align = ['left', 'center'],
                                font = dict(color = 'darkslategray', size = 13.5)
                                ))

   

    trace_piechart = go.Pie(labels=labels, values=values)

    # trace for pie_chart , here extra parameters are just to provide style
    trace_piechart = go.Pie(labels =  labels ,
                  values = values ,  hoverinfo='label+percent', textinfo='value', textfont_size=20 ,
                  marker=dict( line=dict(color='#000000', width=3)) )


    # Merging the given traces[trace_table , trace_pie_chart] using make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "Table"}, {"type": "pie"}]])
    fig.add_trace(trace_table, row=1, col=1)
    fig.add_trace(trace_piechart, row=1, col=2)

    # Updating the size of figure
    fig.update_layout(
    autosize=False,
    width =  1000,
    height = 470)


    return fig