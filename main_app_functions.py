from streamlit import *
from Home_Page_Functions import *
from EDA_Page_Functions import *

# Extra
from models import *

#############################################################################################################################################################################################
def Home():
    Markdown_Style("Feature Engineering" , 3)
    text("")
    text("")
    Markdown_Style("Upload the dataset!!!" , 2)

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
       
        # DataTypes of different features of df
        markdown_type_2 = "Categories of Features : "
        Cool_Data_Printer(markdown_type_2 = markdown_type_2 , write_this = type_of_feature(df))

        # Missing values counter
        markdown_type_2 = "The Missing Values and Strategey :"
        Cool_Data_Printer(markdown_type_2 = markdown_type_2 , write_this = null_value(df))
            
        # Visualising the missing values
        markdown_type_1 = "Heatmap for null values"
        plot = heatmap_generator(df.isnull()) 
        Cool_Plot_Printer(plot , markdown_type_1 = markdown_type_1 , plot_print_type ='pyplot' )

        # Finding imbalanced features 
        imbalanced_features_manager(df)

        # Preparing a lis of categorical feature named categorical and  new_cat[will be used in dropdowns]
        categorical = cat_num(df)
        new_cat = ["Choose The Feature"]
        new_cat.extend(categorical)

        # Pie chart for value_counts of a particular feature
        checkbox_text    = "Show value count of a Categorical feature"
        drop_down_list   =  new_cat
        select_box_text  = "Select Categorical Feature"
        sub_header       = "Categorical feature value counter"
        
        Cool_Data_Plotter(df ,
                          checkbox_text  , 
                          drop_down_list  , 
                          plot_type = 'pie_chart', 
                          sub_header = sub_header , 
                          select_box_text_type_1 = select_box_text  )

        # Two categorical features comparator
        checkbox_text       = "Show compaerison b/w two categorical features"
        drop_down_list      =  new_cat
        select_box_text_lis = ["Select First Categorical Feature" , "Select Second Categorical Feature" ] 
        sub_header          = "Two features categorical values combined comparator"

        Cool_Data_Plotter(df ,
                          checkbox_text ,
                          drop_down_list  ,
                          plot_type = 'comparison_plot',
                          sub_header = sub_header ,
                          select_box_text_type_2 = select_box_text_lis )

        # Feature Dropper
        feature_tracker = feature_dropper(df)

        # Missing values filling system
        missing_values_filling_system(df , feature_tracker)

        # Useless features management system
        flag = useless_features_manager(df)

        # final summary provider
        final_summary_provider(df , flag)


#############################################################################################################################################################################################

        
def EDA():
    Markdown_Style('Exploratory data analysis' , 3)
    text("")
    text("")
    df = pd.read_csv('update.csv')
    dataframe(df.head())

    # Heatmap 
    EDA_heatmap(df)

    # Boxplot
    EDA_boxplot(df)

    # Histogram
    EDA_histogram(df)

    # Sunburst
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

