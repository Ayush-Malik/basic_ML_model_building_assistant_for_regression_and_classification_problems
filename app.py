from streamlit import *
import pandas as pd 
import numpy as np 
from feature_eng import *
from models import *
from EDA import *
import base64


activities = ["Home", "EDA", "Model Building", "About Us"]	
choice = sidebar.selectbox("Select Option",activities)
set_option('deprecation.showfileUploaderEncoding', False)

if choice == "Home": # For Navigating to Home Page
    markdown("<h1 style='text-align: center; color: green;'>Feature Engineering</h1>", unsafe_allow_html=True)
    text("")
    text("")


  
    subheader("Upload the dataset!!!")
    data = file_uploader("" , type=["csv"]) # Loading the dataset


    if data != None : # Here if block runs only when user gives dataset
        df = pd.read_csv(data)
        
        text("")
        text("")  

        subheader("Head the Dataset : ")
        text("")

        dataframe(df.head())
        text("")

        subheader("Shape of the Dataset : ")
        text("")

        write(df.shape)
        text("")
        text("")

        dic = type_of_feature(df)
        subheader("Categories of different Features are : ")
        write(dic)
        text("")

        missing_values_count = null_value(df)
        subheader("The Missing Values In Dataset and Strategey to Fill them : ")
        write(missing_values_count)
        text("")
        text("")

        write("Null Values in Heatmap Form")
        text("")
    
        heat_plot = heatmap_generator(df.isnull())
        pyplot()
        text("")
        text("")

        subheader("Imbalanced Features in Dataset are : ")
        ls = imbalanced_feature(df)

        if ls == []:
            info("There are no imbalanced Features in Dataset")
        else:
            dataframe(ls)
        text("")
        text("")

        categorical = cat_num(df)
        new_cat = ["Choose The Feature"]
        new_cat.extend(categorical)

        if checkbox("Show value count of a Categorical feature"):
            categorical_feature = selectbox("Select Categorical Feature", new_cat)
            if categorical_feature != "Choose The Feature":
                percent_pie = prcntage_values( categorical_feature, df)
                plotly_chart(percent_pie)
            
        if checkbox("Show compaerison b/w two categorical features"):
            subheader("Two features categorical values combined comparator")

            categorical1 = selectbox("Select First Categorical Feature", new_cat)
            categorical2 = selectbox("Select Second Categorical Feature", new_cat)
            if categorical1!= "Choose The Feature" and categorical2 != "Choose The Feature":
                cat_lis          = [categorical1, categorical2] 
                comparison_plot = two_cat_comparator( cat_lis , df )
                plotly_chart(comparison_plot)

        subheader("Select the feature to be dropped")
        missing_lis = missing_value_lis(df)
        lis_drop = multiselect("Select Feature", missing_lis) 
        feature_tracker, sent = drop_feat(df, lis_drop)
        if lis_drop != []:
            success(sent)
        
        subheader("Select Features to be filled")
        lis_fill = []
        feature_ch = []
        count = 0
        for feature in feature_tracker:
            if checkbox(feature):
                strategy = selectbox("Choose strategy", ["strategy","mean", "median", "mode"], key = count)
                if strategy != "strategy":
                    lis_fill.append(strategy)
                    feature_ch.append(feature)
                    success("Feature filled Successfully")
                count += 1
        no_null = fill_feature(df, feature_ch , lis_fill)
        text("")
        write(no_null)

        text("")
        subheader("Checking Useless Features")
        write("The features which have unique values nearly equal to the dataset are:")
        usl_df = useless_feat(df)
        write(usl_df)
        text("")
        flag = 0
        for feature in usl_df["Feature"]:
            if checkbox("Select to drop " + feature):
                drop_useless_feat(df, feature)
                success("Feature Dropped Successfully")
                flag += 1
        text("")
        text("")
        if flag != 0:
            subheader("After Doing all of the above Feature Engineering The datest is now as below")
            text("")
            dataframe(df.head())

            text("")
            write("There are now no null values and also there are no Imbalanced or useless Features")
            heat_plot = heatmap_generator(df.isnull())
            pyplot()

            text("")
            success("Congrats Feature Engineering is Done 🎉🎉. Now You can move to next part, i.e , Doing EDA")

            text("")
            balloons()
        info("To download this updated dataset click the link below")
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        markdown(href, unsafe_allow_html=True)
        df.to_csv("update.csv",index=False)
            

elif choice == "EDA": # For Navigating to Home Page
    markdown("<h1 style='text-align: center; color: green;'>Exploratory data analysis</h1>", unsafe_allow_html=True)
    text("")
    text("")
    df = pd.read_csv('update.csv')
    dataframe(df.head())
    text("")
    numerical_feat,categorical_features = num_num(df)
    if checkbox("Select to Visulaize Correlation heatmap"):
        selected_features = multiselect("Select Feature", numerical_feat)
        if selected_features != []:
            fig = correlation_heatmap(df , selected_features)
            plotly_chart(fig)
    
    text("")
    if checkbox("Select to Visulaize Box Plot"):
        selected_features = multiselect("Select Feature", numerical_feat, key=2)
        if len(selected_features) >= 2:
            fig2 = box_plot(df , selected_features)
            plotly_chart(fig2)

    tot_lis = numerical_feat.copy()
    tot_lis.extend(categorical_features)

    text("")
    if checkbox("Select to Visulaize Histo Gram"):
        selected_features = multiselect("Select Feature", tot_lis, key=4)
        if selected_features != [] and len(selected_features) <= 2:
            fig3 = histo_gram(df , selected_features)
            plotly_chart(fig3)
        elif len(selected_features) > 2:
            warning("You are trying to select excessive Features")

    new_num = numerical_feat.copy()
    newl = ["Feature"]
    newl.extend(new_num)

    text("")
    if checkbox("Select to Visulaize Sun Burst Plot"):
        selected_features = multiselect("Select Feature", tot_lis, key=5)
        text("")
        vals = selectbox("Select Feature", newl)
        if selected_features != [] and vals != "Feature":
            fig4 = sun_burst(df , selected_features, vals)
            plotly_chart(fig4)
