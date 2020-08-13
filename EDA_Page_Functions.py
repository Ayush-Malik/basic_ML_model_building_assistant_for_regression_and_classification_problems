from EDA import *
from streamlit import *



#############################################################################################################################################################################################


def EDA_heatmap(df):

    text("")
    numerical_feat,categorical_features = num_num(df)

    if checkbox("Select to Visulaize Correlation heatmap"):
        selected_features = multiselect("Select Feature", numerical_feat)
        if selected_features != []:
            fig = correlation_heatmap(df , selected_features)
            plotly_chart(fig)


#############################################################################################################################################################################################

def EDA_boxplot(df):

    text("")
    numerical_feat,categorical_features = num_num(df)

    if checkbox("Select to Visulaize Box Plot"):  #must show an error that you cannot pass more than 2 values
        selected_features = multiselect("Select minimum two Feature", numerical_feat,key=2)
        if len(selected_features) >= 2:
            fig2 = box_plot(df , selected_features)
            plotly_chart(fig2)


def EDA_histogram(df):

    text("")
    numerical_feat,categorical_features = num_num(df)

    tot_lis = numerical_feat.copy()
    tot_lis.extend(categorical_features)
    text("")

    if checkbox("Select to Visualize Histo Gram"):
        selected_features = multiselect("Select Feature", tot_lis, key=4)
        if selected_features != [] and len(selected_features) <= 2:
            fig3 = histo_gram(df , selected_features)
            plotly_chart(fig3)
        elif len(selected_features) > 2:
            warning("You are trying to select excessive Features")


def  EDA_sunburst(df):

    text("")
    numerical_feat,categorical_features = num_num(df)

    tot_lis = numerical_feat.copy()
    tot_lis.extend(categorical_features)

    new_num = numerical_feat.copy()
    newl = ["Feature"]
    newl.extend(new_num)

    text("")
    if checkbox("Select to Visualize Sun Burst Plot"):
        selected_features = multiselect("Select Feature", tot_lis, key=5)
        text("")
        vals = selectbox("Select Feature", newl)
        if selected_features != [] and vals != "Feature":
            fig4 = sun_burst(df , selected_features, vals)
            plotly_chart(fig4)

