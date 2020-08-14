from streamlit import *
from main_app_functions import *
# from PIL import Image

markdown_style_sidebar = "text-align: center; font-family: Georgia, Times, serif; font-weight: bolder; font-size:40px; padding-top: 20px; background-image: linear-gradient(to left, rgb(184, 48, 184), rgb(59, 9, 95), blue); - webkit-background-clip: text; - moz-background-clip: text; background-clip: text; color: transparent; "
markdown_style_h       = "font-size:30px; color:green; font-family: Brush Script MT;"


activities = ["Home", "EDA", "Model Building", "About Us"]	

# sidebar.header("ML Automater")
sidebar.markdown("<p style='" + markdown_style_sidebar +"' >" + "ML Automator" + "</p>", unsafe_allow_html=True)
sidebar.text("")

choice = sidebar.selectbox("Select Option",activities)

for i in range(10):
    sidebar.text("") 


sidebar.markdown("<a style = 'font-size : 25px; color : rgb(0 , 0 , 0); position: relative; left: 50px;'  href='https://github.com/Ayush-Malik/basic_ML_model_building_assistant_for_regression_and_classification_problems' target='_blank'>Git Hub link</a>", unsafe_allow_html=True)
sidebar.text("") 

sidebar.markdown("<p style='" + markdown_style_h +"' >" + "Developed by : " + "</p>", unsafe_allow_html=True)
sidebar.text("") 

sidebar.markdown("<a style = 'font-size : 25px; color : rgb(0 , 0 , 0); position: relative; left: 50px;'  href='https://www.linkedin.com/in/ayush-malik-2252b7199/' target='_blank'>Ayush Malik{NOOB}</a>", unsafe_allow_html=True)
sidebar.text("") 

sidebar.markdown("<a style = 'font-size : 25px; color : rgb(0 , 0 , 0); position: relative; left: 50px;'  href='https://github.com/Ayush-Malik/basic_ML_model_building_assistant_for_regression_and_classification_problems' target='_blank'>Git Hub link</a>", unsafe_allow_html=True)
sidebar.text("") 

sidebar.markdown("<a style = 'font-size : 25px; color : rgb(0 , 0 , 0); position: relative; left: 50px;'  href='https://github.com/Ayush-Malik/basic_ML_model_building_assistant_for_regression_and_classification_problems' target='_blank'>Git Hub link</a>", unsafe_allow_html=True)
sidebar.text("") 
# image = Image.open('git_icon.png')
# sidebar.favicon( image ) 

set_option('deprecation.showfileUploaderEncoding', False)

if choice == "Home": # For Navigating to Home Page
    Home()

elif choice == "EDA": # For Navigating to EDA Page
    EDA()

elif choice == 'Model Building': # For Navigating to Model Building page
    Model_Builder()