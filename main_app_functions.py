from streamlit import *
from Home_Page_Functions import *
from EDA_Page_Functions import *

# Extra
from models import *

markdown("<link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'>\
  <script src='https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js></script>\
  <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js'></script>\
  <script src='https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js'></script>", unsafe_allow_html=True)

#############################################################################################################################################################################################
def Home():
    Markdown_Style("Feature Engineering" , 3)
    text("")
    text("")
    Markdown_Style("UPLOAD THE DATASET !!!" , 2)

    data = file_uploader("" , type=["csv"]) # Loading the dataset

    if data is not None : # Here if block runs only when user gives dataset

        #Loading the dataset using pandas
        df = pd.read_csv(data)

        # Head (Top 5 rows) of the dataset
        markdown_type_2 = "Head of the Dataset :"
        Cool_Data_Printer( markdown_type_2 = markdown_type_2 , data_frame = df.head() )

        # Shape (rows x columns) of the dataset
        markdown_type_1 = "Shape of the Dataset : " + str(df.shape)
        Cool_Data_Printer( markdown_type_1 = markdown_type_1 )
       

        # Features Overview provider --> This function provides a table showing datatype of all features + a pie chart showing %age of numerical and categorical features
        markdown_type_2 = "Categories of Features : "
        Cool_Data_Printer(markdown_type_2 = markdown_type_2 )
        plot = features_overview_provider(df)
        plotly_chart( plot )

    
        # Missing values counter
        markdown_type_2 = "The Missing Values and Strategey :"
        Cool_Data_Printer(markdown_type_2 = markdown_type_2 , write_this = null_value(df))
            

        # Visualising the missing values
        markdown_type_1 = "Heatmap for null values"
        plot = heatmap_generator(df) 
        Cool_Plot_Printer(plot , markdown_type_1 = markdown_type_1 , plot_print_type ='plotly_chart' )


        # Finding imbalanced features 
        imbalanced_features_manager(df)

        # Preparing a lis of categorical feature named categorical and  new_cat[will be used in dropdowns]
        categorical = cat_num(df)
        new_cat = ["Choose The Feature"]
        new_cat.extend(categorical)

        # Pie chart for value_counts of a particular feature
        checkbox_text           = "Show value count of a Categorical feature"
        drop_down_list          =  new_cat
        select_box_text_type_1  = ""
        markdown_type_2         = "Categorical feature value counter"
        
        Cool_Data_Plotter(df ,
                          checkbox_text  , 
                          drop_down_list  , 
                          plot_type = 'pie_chart', 
                          markdown_type_2 = markdown_type_2 , 
                          select_box_text_type_1 = select_box_text_type_1  )

        # Two categorical features comparator

        
        checkbox_text       = "Show comparison b/w two categorical features"
        drop_down_list      =  new_cat
        select_box_text_lis = ["" , "" ] 
        markdown_type_2     = "2 Categorical Feature Comparator"

        Cool_Data_Plotter(df ,
                          checkbox_text ,
                          drop_down_list  ,
                          plot_type = 'comparison_plot',
                          markdown_type_2 = markdown_type_2 ,
                          select_box_text_type_2 = select_box_text_lis )

        # Feature Dropper
        feature_tracker = feature_dropper(df)

        # Missing values filling system
        missing_values_filling_system(df , feature_tracker)

        # Useless features management system
        useless_features_manager(df)

        # final summary provider
        if (len(useless_feat(df)) == 0) and (sum(df.isnull().sum()) == 0) : # final summary will be provided only when there's no useless features and no null values in the dataframe
            final_summary_provider(df)


#############################################################################################################################################################################################

        
def EDA():
    Markdown_Style('Exploratory data analysis' , 3)
    text("")
    text("")
    df = pd.read_csv('update.csv')
    dataframe(df.head())
    text("")
    text("")

    # Heatmap 
    Markdown_Style('Correlation Heatmaps' , 2)
    EDA_heatmap(df)
    text("")

    # Boxplot
    Markdown_Style('box type plot' , 2)
    EDA_boxplot(df)
    text("")

    # Histogram
    Markdown_Style('histogram plot' , 2)
    EDA_histogram(df)
    text("")

    # Sunburst
    Markdown_Style('Sun Burst Plot' , 2)
    EDA_sunburst(df)


#############################################################################################################################################################################################


def Model_Builder():
    Markdown_Style('Model Building And Training' , 3)

    text("")
    text("")

    # Loading the updated dataset which ready for model
    df = pd.read_csv('update.csv')

    dataframe(df.head())

    fe_list = feature_list(df)
    text("")
    text("")


    # Getting dummy variables of Categorical features
    df = pd.get_dummies(df , drop_first = True)
    info("After converting Categorical Features into Numerical Ones, The current dataset is")
    dataframe(df.head())
    text("")
    Markdown_Style("Shape of the Dataframe " + str(df.shape), 1)

    # Getting the target feature from the user 
    text("")
    nlist = ["Feature"]
    nlist.extend(fe_list)
    Markdown_Style('Select The Target Feature :' , 2)

    # Suggesting the type of problem (classification/Regression) 
    text("")
    target_feature = selectbox("", nlist)
    if target_feature != "Feature":
        typ, statem = set_target(df, target_feature)
        info(statem)
        if checkbox("Want to change the Problem Type (Classification / Regression)"):
            if typ == "Regression":
                typ = "Classification"
                info("Changed successfully, Now its a " + typ + " Problem")
            else:
                typ = "Regression"
                info("Changed successfully, Now its a " + typ + " Problem")


    # Splitting the dataset into training and testing data
    if target_feature != "Feature":
        text("")
        Markdown_Style("Let's Start Splitting The Dataset" , 2)
        text("")

        prcntage = 0.82
        info("By default Percentage of Training Data is 82 %")
        text("")
        if checkbox("Select to Change Training Dataset Size"):
            prcntage = slider('Select percentage',60, 86)
            write("You selected : ", prcntage, "percent for training Dataset")
            prcntage = prcntage/100
        train, test = train_test_splitter(df, prcntage)

        # Updated shape of training and testing data
        text("")
        Markdown_Style("Shape of the Training Dataset: "+ str(train.shape) , 1)
        text("")
        Markdown_Style("Shape of the Test Dataset:-  "+ str(test.shape) , 1)
        text("")
        text("")

        x_train, x_test, y_train, y_test = x_y_maker(target_feature, train, test)
        info("Now the Train and test dataset are splitted into x_train, x_test, y_train, y_test")

        text("")
        text("")
        x_list = [x_train, x_test]
        y_list = [y_train, y_test]
        mlists = []

        # Taking the models from user which will be used for training
        text("")
        Markdown_Style("Let's Start Model Training", 2)
        text("")
        if typ == "Regression":
            mlists = ['LinearRegression','RandomForestRegressor','AdaBoostRegressor','SVR','MLPRegressor','DecisionTreeRegressor','XGBRegressor']
        else:
            mlists = ['LogisticRegression','RandomForestClassifier','AdaBoostClassifier','SVC','MLPClassifier','DecisionTreeClassifier','XGBClassifier']
        models_lists = multiselect("Select Models", mlists)
        model_object = Models(x_list, y_list, typ,models_lists)
        model_object.model_call()
        extra = ["Select"]
        extra.extend(models_lists)

        # Predictions Downloader
        text("")
        text("")
        Markdown_Style("Y_Pred Dataset Downloader", 2)
        text("")
        if checkbox("Select to get Y_predictions of a model"):
            selectd_models = selectbox("Select Model", extra)
            if selectd_models != "Select":
                y_pred = model_object.output(selectd_models)
                y_pred = pd.DataFrame(y_pred)
                dataframe(y_pred)
                text("")
                csv = y_pred.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                markdown(href, unsafe_allow_html=True)

#############################################################################################################################################################################################
def name_styler(name):
    name = "<h2 style='color:rgb(44, 52, 84);font-size:34px;font-weight:800'><u>" + name + "</u></h2>"
    return name

def image_maker(image_name):
    with open(image_name, "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode()
        imigi = "<img style='height400px;' src='data:image/png;base64,{}' class='img-fluid'>".format(my_string)
        return imigi

def About_Us():
    Markdown_Style("about the website", 2)
    text("")
    text("")

    # Info About Web App
    markdown("<p style='margin-left:30px;'>Our site provides Feature Engineering Tools, Expolaratory Data Analalysis, Machine learning model building and training in a very easy and automated way.\
        It's Basically a web app built using Streamlit module that provides effective solution to most of the machine learning usecases. It actually takes the dataset from user and based on \
        user interest it first does the Feature Engineering and after that user can do EDA and after doing all that user can start building model building.<br> <b><i>Note :-Currently It only provides the solution for Classification and Regression Problem</i></b></p>", unsafe_allow_html=True)
    text("")
    text("")

    Markdown_Style("about the developers", 2)
    text("")
    text("")

    # About 1st Developer --- Ayush Malik            
    image1 = image_maker("ayush_about.jpeg")
    markdown("<div class='card' style='width:370px'>"\
    + image1 + \
    "<div style='background-color:#a6f5dd' class='card-body'>"\
      + name_styler('Ayush Malik') +\
      "<p class='card-text'>ML Entusiast | Python & Django Developer | Jmitian</p>\
      <a href='https://www.linkedin.com/in/ayush-malik-2252b7199/'  target='_blank' class='btn btn-success' style='color:black; font-weight:500'>Linkdin Profile</a>\
    <a href='https://github.com/Ayush-Malik' class='btn btn-success'  target='_blank' style='color:black; font-weight:500'>Github Profile</a>\
    </div>\
    </div>", unsafe_allow_html=True)

    # About 2nd Developer --- Abhay Dhiman
    image2 = image_maker("abhay_about.jpeg")
    markdown("<div  class='card' style='width:370px; margin-left:400px; '>"\
    + image2 + \
    "<div style='background-color:#a6f5dd' class='card-body'>"\
      + name_styler('Abhay Dhiman') +\
      "<p class='card-text'>ML & Deep Learning Entusiast | Python Developer | Jmitian</p>\
      <a href='https://www.linkedin.com/in/abhay-dhiman-409378191/'  target='_blank' class='btn btn-success' style='color:black; font-weight:500'>Linkdin Profile</a>\
    <a href='https://github.com/abhaydhiman' class='btn btn-success'  target='_blank' style='color:black; font-weight:500'>Github Profile</a>\
    </div>\
    </div>", unsafe_allow_html=True)

    # About 3rd Developer --- Aaditya Singhal
    image3 = image_maker("my image.jpg")
    markdown("<div  class='card' style='width:370px;'>"\
    + image3 + \
    "<div style='background-color:#a6f5dd' class='card-body'>"\
      + name_styler('Aaditya Singhal') +\
      "<p class='card-text'>ML & Deep Learning Entusiast | Python & flask Developer | Jmitian</p>\
      <a href='https://www.linkedin.com/in/aaditya-singhal-a46720192/'  target='_blank' class='btn btn-success' style='color:black; font-weight:500'>Linkdin Profile</a>\
    <a href='https://github.com/Aaditya1978' class='btn btn-success'  target='_blank' style='color:black; font-weight:500'>Github Profile</a>\
    </div>\
    </div>", unsafe_allow_html=True)
