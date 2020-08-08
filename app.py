import streamlit as st
import pandas as pd 
import numpy as np 
from feature_eng import *


activities = ["Home", "EDA", "Model Building", "About UsS"]	
choice = st.sidebar.selectbox("Select Option",activities)
st.set_option('deprecation.showfileUploaderEncoding', False)

if choice == "Home":
    st.markdown("<h1 style='text-align: center; color: green;'>Exploratory data analysis</h1>", unsafe_allow_html=True)
    st.text("")
    st.text("")
    data = st.file_uploader("Upload The Dataset", type=["csv"])
    if data != None :
        df = pd.read_csv(data)
        st.text("")
        st.text("")        
        st.subheader("Head the Dataset : ")
        st.text("")
        st.dataframe(df.head())
        st.text("")
        st.subheader("Shape of the Dataset : ")
        st.text("")
        st.write(df.shape)
        st.text("")
        st.text("")
        dic = type_of_feature(df)
        st.subheader("Categories of different Features are : ")
        st.write(dic)
        st.text("")
        missing_values_count = null_value(df)
        st.subheader("The Missing Values In Dataset and Strategey to Fill them : ")
        st.write(missing_values_count)
        st.text("")
        st.text("")
        st.write("Null Values in Heatmap Form")
        st.text("")
        heat_plot = heatmap_generator(df.isnull())
        st.pyplot()
        st.text("")
        st.text("")
        st.subheader("Imbalanced Features in Dataset are : ")
        ls = imbalanced_feature(df)
        if ls == []:
            st.write("There are no imbalanced Features in Dataset")
        else:
            st.dataframe(ls)
        st.text("")
        st.text("")
        categorical = cat_num(df)
        categorical_feature = st.selectbox("Select Categorical Feature", categorical)
        percent_pie = prcntage_values( categorical_feature, df)
        st.write(percent_pie)
        st.pyplot()
        st.subheader("Two features categorical values combined comparator")
        categorical1 = st.selectbox("Select First Categorical Feature", categorical)
        categorical2 = st.selectbox("Select Second Categorical Feature", categorical)
        cat_lis = [categorical1, categorical2]
        comp_plot = different_cat_comparator(cat_lis, df)
        st.write(comp_plot)
        st.pyplot()

       