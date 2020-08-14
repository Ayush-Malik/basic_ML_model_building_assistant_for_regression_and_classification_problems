from streamlit import *
from main_app_functions import *

activities = ["Home", "EDA", "Model Building", "About Us"]	
choice = sidebar.selectbox("Select Option",activities)
set_option('deprecation.showfileUploaderEncoding', False)

if choice == "Home": # For Navigating to Home Page
    Home()

elif choice == "EDA": # For Navigating to EDA Page
    EDA()

elif choice == 'Model Building': # For Navigating to Model Building page
    Model_Builder()