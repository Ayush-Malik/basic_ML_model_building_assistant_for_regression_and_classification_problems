from streamlit import *
import pandas as pd 
import numpy as np 
from feature_eng import *
import base64
#############################################################################################################################################################################################


# Markdown Styles
markdown_style1 = "position: relative; left: 50px; font-size:30px; color:grey; font-family: Brush Script MT;"
markdown_style2 = "position: relative; font-size:30px; color:brown; font-family: Algerian;"
markdown_style3 = "text-align: center; font-family: Georgia, Times, serif; font-weight: bolder; font-size:40px; padding-top: 20px; background-image: linear-gradient(to left, rgb(184, 48, 184), rgb(59, 9, 95), blue); - webkit-background-clip: text; - moz-background-clip: text; background-clip: text; color: transparent; "

def Markdown_Style(value , type = 1):
    if type == 1:
        style_type = markdown_style1

    elif type == 2:
        style_type = markdown_style2
    
    elif type == 3:
        style_type = markdown_style3


    markdown("<p style='" + style_type +
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
        Markdown_Style( markdown_type_1 , 1 )

    if markdown_type_2 is not None:
        Markdown_Style( markdown_type_2 , 2)
    
    if data_frame is not None:
        dataframe( data_frame.head() )
    
    if write_this is not None:
        write( write_this )
    
    if print_info is not None:
        info(print_info)


#############################################################################################################################################################################################


''' This particular function is used to print plots using 'pyplot' 
    or 'plotly_chart' functions of streamlit with a markdown_type_1 '''
def Cool_Plot_Printer(plot , sub_header = None , markdown_type_1 = None , markdown_type_2 = None  , markdown_type_3 = None , plot_print_type = None  ):

    if sub_header is not None:
        subheader( sub_header )

    if markdown_type_1 is not None:
        Markdown_Style( markdown_type_1 , 1 )

    if markdown_type_2 is not None:
        Markdown_Style( markdown_type_2 , 2 )
    
    if markdown_type_3 is not None:
        Markdown_Style( markdown_type_3 , 3 )
    
    if plot_print_type == 'pyplot':
        pyplot()
    elif plot_print_type == 'plotly_chart':
        plotly_chart( plot )


#############################################################################################################################################################################################


''' This particular function provides [ checkbox + selectbox / Multiselect ] 
    functionalitis and it return , it returns a list of features selected by user '''
def Cool_Data_Plotter(df , checkbox_text , drop_down_list  , plot_type , sub_header = None , markdown_type_1 = None , markdown_type_2 = None , markdown_type_3 = None ,    select_box_text_type_1 = None  , select_box_text_type_2 = None , multi_select_box_text = None   ):
    
    if sub_header is not None:
        subheader( sub_header )

    if markdown_type_1 is not None:
        Markdown_Style( markdown_type_1 , 1 )

    if markdown_type_2 is not None:
        Markdown_Style( markdown_type_2 , 2 )
    
    if markdown_type_3 is not None:
        Markdown_Style( markdown_type_3 , 3 )
    
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


def imbalanced_features_manager(df):
    ls = imbalanced_feature(df)
    if ls == []:
        info("There are no imbalanced Features in Dataset")
    else:
        Markdown_Style("Imbalanced Features in Dataset are : " , 2)
        dataframe(ls)
    text("")
    text("")


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
        if button("Click if All done"):
            subheader("After Doing all of the above Feature Engineering The dataset is now as below")
            text("")
            dataframe(df.head())

            text("")
            write("There are now no null values and also there are no Imbalanced or useless Features")

            text("")
            success("Congrats Feature Engineering is Done 🎉🎉. Now You can move to next part, i.e , Doing EDA")

            text("")
            balloons()
        info("To download this updated dataset click the link below")
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        markdown(href, unsafe_allow_html=True)
        df.to_csv("update.csv",index=False)
