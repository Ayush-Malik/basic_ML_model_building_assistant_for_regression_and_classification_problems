from streamlit import *
from main_app_functions import *
import base64
import os


link2 = "<link href='https://fonts.googleapis.com/css2?family=Lato:ital,wght@1,700&display=swap' rel='stylesheet'>"
markdown_style_sidebar = "text-align: center; font-family: Georgia, Times, serif; font-weight: bolder; font-size:40px; padding-top: 20px; background-image: linear-gradient(to left, rgb(0, 179, 60), rgb(0, 179, 134), blue); - webkit-background-clip: text; - moz-background-clip: text; background-clip: text; color: transparent; "
markdown_style_h1      = "font-size:40px; color:black; font-family: lato, sans-serif;"
markdown_style_h2      = "font-family: lato, sans-serif; font-size: 20px; font-variant: normal; font-weight: 700; line-height: 15.4px; position:relative; left:30px; color:grey"


set_option('deprecation.showfileUploaderEncoding', False)
activities = ["Home", "EDA", "Model Building", "About Us"]	


sidebar.markdown("<p style='" + markdown_style_sidebar +"' >" + "ML Automator" + "</p>", unsafe_allow_html=True)
sidebar.text("")
sidebar.text("")

choice = sidebar.selectbox("Select Option",activities)

# for i in range(10):
#     sidebar.text("") 

def image_maker(image_name):
    with open(image_name, "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode()
        imigi = "<img style='position:fixed; left:1px; bottom:1px; width:60px; height:60px; border-radius: 50%;' src='data:image/png;base64,{}' class='img-fluid'>".format(my_string)
        return imigi

image1 = image_maker("git_icon.png")

sidebar.markdown(
    "<a style='font-size:20px; color:rgb(104, 96, 96); position:relative; left:30px;'  href='https://github.com/Ayush-Malik/basic_ML_model_building_assistant_for_regression_and_classification_problems/tree/streamlit_autoML' target='_blank'>" + image1 +"</a>",
    unsafe_allow_html=True
    )
sidebar.text("") 
sidebar.text("")
sidebar.text("")

image2 = image_maker("my image.jpg")
image3 = image_maker("ab_image.jfif")
image4 = image_maker("ay_image.jfif")

sidebar.markdown(link2 + "<p style='" + markdown_style_h1 +"' >" + "Developed by : " + "</p>", unsafe_allow_html=True)
sidebar.text("") 

# sidebar.markdown(image4 + link2 + "<a style='" + markdown_style_h2 + "' " +"href='https://www.linkedin.com/in/ayush-malik-2252b7199/' target='_blank'>Ayush Malik</a>", unsafe_allow_html=True)
# sidebar.text("") 

# sidebar.markdown(image3 +link2 +  "<a style='" + markdown_style_h2 + "' " +"href='https://www.linkedin.com/in/abhay-dhiman-409378191/' target='_blank'>Abhay Dhiman</a>", unsafe_allow_html=True)
# sidebar.text("") 

# sidebar.markdown(image2 +link2 +  "<a style='" + markdown_style_h2 + "' " +"href='https://www.linkedin.com/in/aaditya-singhal-a46720192/' target='_blank'>Aaditya Singhal</a>", unsafe_allow_html=True)

# for i in range(4):
#     sidebar.text("") 


sidebar.markdown("<p style='position: static; left:200px; bottom:0px; font-size:20px; font-weight:800;'>@Pro_Coders</p>", unsafe_allow_html=True)


# sidebar.markdown("<p>Made With ❤ @Pro_Coders</p>", unsafe_allow_html=True)


if choice == "Home": # For Navigating to Home Page
    Home()

elif choice == "EDA": # For Navigating to EDA Page only when there's no null values in dataframe
    
    Markdown_Style('Exploratory data analysis' , 3)
    text("")
    text("")

    df = pd.read_csv('update.csv')

    if sum( df.isnull().sum() ) != 0:
        subheader("Oops something went wrong , looks like there are Null values in DataFrame")
        text("")
        text("")
        null_df = df.isnull().sum().sort_values( ascending = False).to_frame().reset_index().rename(columns = {'index' : 'Feature' , 0 : 'Null val count'})
        dataframe( null_df  , width = 1000 , height = 1000  )

    else:
        EDA()

elif choice == 'Model Building': # For Navigating to Model Building page only when there's no null values in df
    
    Markdown_Style('Model Building And Training' , 3)
    text("")
    text("")

    df = pd.read_csv('update.csv')

    if sum( df.isnull().sum() ) != 0:
        subheader("Oops something went wrong , looks like there are Null values in DataFrame")
        text("")
        text("")
        null_df = df.isnull().sum().sort_values( ascending = False).to_frame().reset_index().rename(columns = {'index' : 'Feature' , 0 : 'Null val count'})
        dataframe( null_df  , width = 1000 , height = 1000  )

    else:    
        Model_Builder()

elif choice == 'About Us': # For Navigating to About Us Page
    About_Us()
