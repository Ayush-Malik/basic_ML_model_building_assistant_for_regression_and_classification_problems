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