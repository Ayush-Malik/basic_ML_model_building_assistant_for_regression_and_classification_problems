import streamlit as st
import pandas as pd 
import numpy as np 

st.subheader("Exploratory Data Analysis")
st.set_option('deprecation.showfileUploaderEncoding', False)
data = st.file_uploader("Upload a Dataset", type=["csv"])
if data != None :
    df = pd.read_csv(data)
    st.dataframe(df.head())