from streamlit import *
import pandas as pd 
import numpy as np 
from feature_eng import *
import base64


#############################################################################################################################################################################################


# Markdown Styles
markdown_style = "position: relative; left: 50px; font-size:30px; color:grey; font-family: Brush Script MT;"
markdown_style2 = "position: relative; font-size:30px; color:brown; font-family: Algerian;"

def Markdown_Style(value , type = 1):
    if type == 1:
        markdown("<p style='" + markdown_style +
                    "' >" + value + "</p>", unsafe_allow_html=True)
    elif type == 2:
        markdown("<p style='" + markdown_style2 +
                    "' >" + value + "</p>", unsafe_allow_html=True)


#############################################################################################################################################################################################
            

''' As the name suggests the given function contains important 
    functionalities of streamlit at a single place '''
def Cool_Data_Printer( sub_header = None , markdown_type_1 = None , markdown_type_2 = None ,  data_frame = None , write_this = None , print_info = None):
    text("")
    text("")
        
    if sub_header is not None:
        subheader( sub_header )

    if markdown_type_1 is not None:
        Markdown_Style( markdown_type_1 , 2 )

    if markdown_type_2 is not None:
        Markdown_Style( markdown_type_2 , 1)
    
    if data_frame is not None:
        dataframe( data_frame.head() )
    
    if write_this is not None:
        write( write_this )
    
    if print_info is not None:
        info(print_info)


#############################################################################################################################################################################################


''' This particular function is used to print plots using 'pyplot' 
    or 'plotly_chart' functions of streamlit with a markdown_type_1 '''
def Cool_Plot_Printer(plot , sub_header = None , markdown_type_1 = None , markdown_type_2 = None , plot_print_type = None  ):

    if sub_header is not None:
        subheader( sub_header )

    if markdown_type_1 is not None:
        Markdown_Style( markdown_type_1 , 2 )

    if markdown_type_2 is not None:
        Markdown_Style( markdown_type_2 , 1)
    
    if plot_print_type == 'pyplot':
        pyplot()
    elif plot_print_type == 'plotly_chart':
        plotly_chart( plot )


#############################################################################################################################################################################################


''' This particular function provides [ checkbox + selectbox / Multiselect ] 
    functionalitis and it return , it returns a list of features selected by user '''
def Cool_Data_Plotter(df , checkbox_text , drop_down_list  , plot_type , markdown_type_1 = None ,  select_box_text_type_1 = None  , select_box_text_type_2 = None , multi_select_box_text = None   ):
    
    if markdown_type_1 is not None:
        subheader( markdown_type_1 )
    
    
    if checkbox( checkbox_text ):

        if select_box_text_type_1 is not None: # Single Checkbox

            categorical_feature = selectbox( select_box_text_type_1 , drop_down_list)

            if categorical_feature != drop_down_list[0]:
                unique_len = len(df[categorical_feature].value_counts())
                if unique_len > 7:
                    dataframe(df[categorical_feature].value_counts())
                    Markdown_Style("Total unique values : " + str(unique_len) , 1)
                elif plot_type == 'pie_chart':
                    percent_pie = prcntage_values( categorical_feature, df)
                    plotly_chart(percent_pie)

        elif select_box_text_type_2 is not None: # Two Checkboxes

            categorical1 = selectbox( select_box_text_type_2[0] , drop_down_list )
            categorical2 = selectbox( select_box_text_type_2[1] , drop_down_list )

            if (categorical1 != drop_down_list[0])  and  (categorical2 != drop_down_list[0]):
                cat_lis          = [categorical1, categorical2] 

                if plot_type == 'comparison_plot':
                    comparison_plot  = two_cat_comparator( cat_lis , df )
                    plotly_chart(comparison_plot)


#############################################################################################################################################################################################


''' This function accepts a df and drop all 
    the features selected by user and returns a feature_tracker
    list which contains active null_val feature '''
def feature_dropper(df):
    markdown_type_1 = "Select the feature to be dropped : "
    Markdown_Style(markdown_type_1 , 2)
    missing_lis = missing_value_lis(df)
    lis_drop = multiselect("Select Feature", missing_lis) 
    feature_tracker, sent = drop_feat(df, lis_drop)
    if lis_drop != []:
        success(sent)
    return feature_tracker


#############################################################################################################################################################################################

        
def missing_values_filling_system(df , feature_tracker):
    Markdown_Style( "Select Features to be filled" , 2)
    lis_fill = []
    feature_ch = []
    count = 0
    for feature in feature_tracker:
        if checkbox(feature):
            strategy = selectbox("Choose strategy", ["strategy","mean", "median", "mode"], key = count)
            if strategy != "strategy":
                lis_fill.append(strategy)
                feature_ch.append(feature)
                success("Feature filled Successfully")
            count += 1
    no_null = fill_feature(df, feature_ch , lis_fill)
    text("")
    write(no_null)




#############################################################################################################################################################################################


def useless_features_manager(df):
        text("")
        Markdown_Style("Useless Features :" , type = 2)
        write("The features which have high unique values are:")
        usl_df = useless_feat(df)
        write(usl_df)
        text("")
        flag = 0
        for feature in usl_df["Feature"]:
            if checkbox('Select to drop ' +  feature):
                drop_useless_feat(df, feature)
                success("Feature Dropped Successfully")
                flag += 1
        text("")
        text("")
        return flag


#############################################################################################################################################################################################


def final_summary_provider(df , flag):
        if flag != 0:
            subheader("After Doing all of the above Feature Engineering The datest is now as below")
            text("")
            dataframe(df.head())

            text("")
            write("There are now no null values and also there are no Imbalanced or useless Features")
            heat_plot = heatmap_generator(df.isnull())
            pyplot()

            text("")
            success("Congrats Feature Engineering is Done 🎉🎉. Now You can move to next part, i.e , Doing EDA")

            text("")
            balloons()
        info("To download this updated dataset click the link below")
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        markdown(href, unsafe_allow_html=True)
        df.to_csv('update.csv' , index=False)


#############################################################################################################################################################################################
def Home():
    markdown("<h1 style='text-align: center; color: green;'>Feature Engineering</h1>", unsafe_allow_html=True)
    text("")
    text("")
    markdown_type_1 = "Upload the dataset!!!"
    Markdown_Style(markdown_type_1 , 2)

    data = file_uploader("" , type=["csv"]) # Loading the dataset

    if data is not None : # Here if block runs only when user gives dataset

        #Loading the dataset using pandas
        df = pd.read_csv(data)

        # Head (Top 5 rows) of the dataset
        markdown_type_1 = "Head of the Dataset :"
        Cool_Data_Printer( markdown_type_1 = markdown_type_1 , data_frame = df.head() )

        # Shape (rows x columns) of the dataset
        markdown_type_2 = "Shape of the Dataset : " + str(df.shape)
        Cool_Data_Printer( markdown_type_2 = markdown_type_2 )
       
        # DataTypes of different features of df
        markdown_type_1 = "Categories of Features : "
        Cool_Data_Printer(markdown_type_1 = markdown_type_1 , write_this = type_of_feature(df))

        # Missing values counter
        markdown_type_1 = "The Missing Values and Strategey :"
        Cool_Data_Printer(markdown_type_1 = markdown_type_1 , write_this = null_value(df))
            
        # Visualising the missing values
        markdown_type_2 = "Heatmap for null values"
        plot = heatmap_generator(df.isnull()) 
        Cool_Plot_Printer(plot , markdown_type_2 = markdown_type_2 , plot_print_type ='pyplot' )

        # # Finding imbalanced features 
        # markdown_type_1 = "Imbalanced Features in Dataset are : "
        # ls = imbalanced_feature(df)
        # if ls == []:
        #     print_info = "There are no imbalanced Features in Dataset..."
        #     Cool_Data_Printer(markdown_type_1 = markdown_type_1 , print_info = print_info )
        # else:
        #     Cool_Data_Printer(markdown_type_1 = markdown_type_1 , data_frame = ls)

        # Preparing a lis of categorical feature named categorical and  new_cat[will be used in dropdowns]
        categorical = cat_num(df)
        new_cat = ["Choose The Feature"]
        new_cat.extend(categorical)

        # Pie chart for value_counts of a particular feature
        checkbox_text    = "Show value count of a Categorical feature"
        drop_down_list   =  new_cat
        select_box_text  = "Select Categorical Feature"
        markdown_type_1       = "Categorical feature value counter"
        
        Cool_Data_Plotter(df ,
                          checkbox_text  , 
                          drop_down_list  , 
                          plot_type = 'pie_chart', 
                          markdown_type_1 = markdown_type_1 , 
                          select_box_text_type_1 = select_box_text  )

        # Two categorical features comparator
        checkbox_text       = "Show compaerison b/w two categorical features"
        drop_down_list      =  new_cat
        select_box_text_lis = ["Select First Categorical Feature" , "Select Second Categorical Feature" ] 
        markdown_type_1          = "Two features categorical values combined comparator"

        Cool_Data_Plotter(df ,
                          checkbox_text ,
                          drop_down_list  ,
                          plot_type = 'comparison_plot',
                          markdown_type_1 = markdown_type_1 ,
                          select_box_text_type_2 = select_box_text_lis )

        # Feature Dropper
        feature_tracker = feature_dropper(df)

        # Missing values filling system
        missing_values_filling_system(df , feature_tracker)

        # Useless features management system
        flag = useless_features_manager(df)

        # final summary provider
        final_summary_provider(df , flag)




        





# def EDA():
#     markdown("<h1 style='text-align: center; color: green;'>Exploratory data analysis</h1>", unsafe_allow_html=True)
#     text("")
#     text("")
#     df = pd.read_csv(csv)
#     write(df.shape)
