from streamlit import *
import pandas as pd 
import numpy as np 
from feature_eng import *


activities = ["Home", "EDA", "Model Building", "About Us"]	
choice = sidebar.selectbox("Select Option",activities)
set_option('deprecation.showfileUploaderEncoding', False)

if choice == "Home": # For Navigating to Home Page
    markdown("<h1 style='text-align: center; color: green;'>Exploratory data analysis</h1>", unsafe_allow_html=True)
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
            write("There are no imbalanced Features in Dataset")
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
                write(percent_pie)
                pyplot()
            
        if checkbox("Show compaerison b/w two categorical features"):
            subheader("Two features categorical values combined comparator")

            categorical1 = selectbox("Select First Categorical Feature", new_cat)
            categorical2 = selectbox("Select Second Categorical Feature", new_cat)
            if categorical1!= "Choose The Feature" and categorical2 != "Choose The Feature":
                cat_lis          = [categorical1, categorical2] 
                comparison_plot = two_cat_comparator( cat_lis , df )
                write(comparison_plot)
                pyplot()



