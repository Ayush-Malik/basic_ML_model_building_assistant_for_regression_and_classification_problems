from streamlit import *
from Home_Page_Functions import *
from EDA_Page_Functions import *

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


