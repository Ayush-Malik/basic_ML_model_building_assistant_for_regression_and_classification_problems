# Importing the necessary libraries
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


# [1] Correlation heatmap plotting function
# --> This particular function accepts a list with selected categorical features and returns a correlation heatmap
def correlation_heatmap(df , selected_features):
    # for example selected_features = ['Age' , 'Fare' , 'Survived'] , This will produce a correlation heatmap of 3x3
    data = np.array( df[ selected_features ].corr() )
    fig = px.imshow(data,
                    labels=dict( color = "Productivity"),
                    x=  lis,
                    y = lis
                )
    return fig


# [2] Box plot function
# --> This particular function accepts either a list of two or three feature but one of the feature must be continuous , in order to get useful visualisation
def box_plot(df , selected_features):

    # for example selected_features = ['Sex' , 'Age' , 'Embarked'] , It will return a box plot where x-axis => Sex , y-axis => Age  , x-axis feature is further categorised on the bases of different features present in 3rd parameter['Embarked' here]
    # Its always better if you pass 1st and 3rd feature of categorical dtype and 2nd of continuous numerical dtype
    # Its good for observing three different features in a single plot 

    if len(selected_features) == 3: # In case when three features are passed
        z = df[selected_features[2]]
    else:
        z = None

    fig = px.box(df, selected_features[0] , selected_features[1] , z )
    fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
    return fig


# [3] Histogram function
# --> This particular function accepts a lis either of size 1 or 2 in format [continuous numerical feature , categorical feature] and returns histogram plot
def histo_gram(df , selected_features ):
    # for example selected_features = ['Age' , 'Embarked']
    if len(selected_features) == 2:
        color = selected_features[1]
    else:
        color = None

    fig = px.histogram(df, x = selected_features[0] , color = color )
    return fig

# [4] Sunburst function
# --> This particuar function accepts a lis of any number of features and return a sun burst plot 
# This is one of the best plotly plot to effectively visualise many features in a single plot
def sun_burst(df , lis_of_features , vals):
    # for example selected_features = ['Sex' , 'Embarked'] , vals = 'Survived
    fig = px.sunburst(df , path = lis_of_features , values = vals)
    return fig




