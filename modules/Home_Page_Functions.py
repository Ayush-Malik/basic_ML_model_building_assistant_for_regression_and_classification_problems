from streamlit import *
import pandas as pd
import numpy as np
from modules.data_preprocessing import *
import base64

#############################################################################################################################################################################################


def Markdown_Style(value, type=1, size=None, border_box=True):
    # Markdown Styles
    length = len(value)
    link = "<link href='https://fonts.googleapis.com/css?family=Anton' rel='stylesheet'>"
    link2 = "<link href='https://fonts.googleapis.com/css2?family=Lato:ital,wght@1,700&display=swap' rel='stylesheet'>"
    markdown_style1 = "position: relative; left: 50px; font-size:24px; color:rgb(14,179,83); font-family: Lato, sans-serif;"

    if size is not None:
        size_code = "font-size: " + str(size) + "px;"
    else:
        size_code = "font-size: 33px;"

    if border_box == True:
        border_code = "border-radius: 25px;border: 8px solid grey;"
    else:
        border_code = ""

    markdown_style2 = "position:relative; font-family:Anton;" + size_code + border_code + \
        "padding: 20px;width: "+str(length*20) + \
        "px;height: 100px;text-align:center;"
    markdown_style3 = "text-align: center; font-family: anton; font-weight: 300; font-size:60px; padding-top: 20px; background-image: linear-gradient(to left, #7c0909, #09477c, green); - webkit-background-clip: text; - moz-background-clip: text; background-clip: text; color: transparent; "

    if type == 1:
        style_type = markdown_style1
        markdown(link2 + "<p style='" + style_type +
                 "' >" + value + "</p>", unsafe_allow_html=True)

    elif type == 2:
        value = value.upper()
        style_type = markdown_style2
        markdown(link + "<p style='" + style_type +
                 "' >" + value + "</p>", unsafe_allow_html=True)

    elif type == 3:
        style_type = markdown_style3
        value = value.upper()
        markdown(link + "<p style='" + style_type +
                 "' >" + value + "</p>", unsafe_allow_html=True)


#############################################################################################################################################################################################
''' As the name suggests the given function contains important 
    functionalities of streamlit at a single place '''


def Cool_Data_Printer(sub_header=None, markdown_type_1=None, markdown_type_2=None,  data_frame=None, write_this=None, print_info=None):
    text("")
    text("")

    if sub_header is not None:
        subheader(sub_header)

    if markdown_type_1 is not None:
        Markdown_Style(markdown_type_1, 1)

    if markdown_type_2 is not None:
        Markdown_Style(markdown_type_2, 2)

    if data_frame is not None:
        dataframe(data_frame.head())

    if (write_this is not None) and (len(write_this) != 0):
        write(write_this)

    if print_info is not None:
        info(print_info)


#############################################################################################################################################################################################


''' This particular function is used to print plots using 'pyplot' 
    or 'plotly_chart' functions of streamlit with a markdown_type_1 '''


def Cool_Plot_Printer(plot, sub_header=None, markdown_type_1=None, markdown_type_2=None, markdown_type_3=None, plot_print_type=None):

    if sub_header is not None:
        subheader(sub_header)

    if markdown_type_1 is not None:
        Markdown_Style(markdown_type_1, 1)

    if markdown_type_2 is not None:
        Markdown_Style(markdown_type_2, 2)

    if markdown_type_3 is not None:
        Markdown_Style(markdown_type_3, 3)

    if plot_print_type == 'pyplot':
        pyplot()
    elif plot_print_type == 'plotly_chart':
        plotly_chart(plot)


#############################################################################################################################################################################################


''' This particular function provides [ checkbox + selectbox / Multiselect ] 
    functionalitis and it return , it returns a list of features selected by user '''


def Cool_Data_Plotter(df, checkbox_text, drop_down_list, plot_type, sub_header=None, markdown_type_1=None, markdown_type_2=None, markdown_type_3=None,    select_box_text_type_1=None, select_box_text_type_2=None, multi_select_box_text=None):

    if sub_header is not None:
        subheader(sub_header)

    if markdown_type_1 is not None:
        Markdown_Style(markdown_type_1, 1)

    if markdown_type_2 is not None:
        Markdown_Style(markdown_type_2, 2)

    if markdown_type_3 is not None:
        Markdown_Style(markdown_type_3, 3)

    if checkbox(checkbox_text):

        if select_box_text_type_1 is not None:  # Single Checkbox

            categorical_feature = selectbox(
                select_box_text_type_1, drop_down_list, key=183737487)

            if categorical_feature != drop_down_list[0]:
                unique_len = len(df[categorical_feature].value_counts())
                if unique_len > 15:
                    dataframe(df[categorical_feature].value_counts())
                    Markdown_Style("Total unique values : " +
                                   str(unique_len), 1)
                elif plot_type == 'pie_chart':
                    # percent_pie = prcntage_values( categorical_feature, df)
                    fig = suplots_maker_for_table_and_piechart(
                        df, type_null=True, feature=categorical_feature)
                    plotly_chart(fig)

        elif select_box_text_type_2 is not None:  # Two Checkboxes

            drop_down_list[0] = "Choose the First Feature"
            first_one = drop_down_list

            categorical1 = selectbox(
                select_box_text_type_2[0], first_one, key=9929389)

            # 2nd selectbox will appear only when 1st is already selected---selectbox -> 1
            if categorical1 != drop_down_list[0]:

                drop_down_list[0] = "Choose the Second Feature"
                second_one = drop_down_list
                second_one.remove(categorical1)

                categorical2 = selectbox(
                    select_box_text_type_2[1], second_one)  # ---selectbox -> 2

                if categorical2 != drop_down_list[0]:
                    cat_lis = [categorical1, categorical2]
                    comparison_plot = two_cat_comparator(cat_lis, df)

                    if plot_type == 'comparison_plot':
                        try:
                            plotly_chart(comparison_plot)
                        except:
                            dataframe(comparison_plot)
                            text("")
                            info("x-axis--> " + categorical1 +
                                 ", y-axis--> " + categorical2)

    text("")
    text("")
    text("")


#############################################################################################################################################################################################


''' This function accepts a df and drop all 
    the features selected by user and returns a feature_tracker
    list which contains active null_val feature '''


def feature_dropper(df):
    markdown_type_1 = "Select the feature to be dropped : "
    Markdown_Style(markdown_type_1, 2)
    missing_lis = missing_value_lis(df)
    lis_drop = multiselect("", missing_lis)
    feature_tracker, sent = drop_feat(df, lis_drop)
    if lis_drop != []:
        success(sent)
    text("")
    text("")
    return feature_tracker


#############################################################################################################################################################################################


def missing_values_filling_system(df, feature_tracker):
    Markdown_Style("Select Features to be filled", 2)
    lis_fill = []
    feature_ch = []
    count = 0
    for feature in feature_tracker:
        if checkbox(feature):
            if df.dtypes[feature] == 'object':
                stratigies_lis = ["strategy",  "mode"]
            else:
                stratigies_lis = ["strategy", "mean", "median"]

            strategy = selectbox("Choose strategy", stratigies_lis, key=count)
            if strategy != "strategy":
                lis_fill.append(strategy)
                feature_ch.append(feature)
                success("Feature filled Successfully")
            count += 1
    no_null = fill_feature(df, feature_ch, lis_fill)
    text("")
    write(no_null)


#############################################################################################################################################################################################

def features_overview_provider(df):
    fig = suplots_maker_for_table_and_piechart(
        df, type_null=False, feature=None)
    return fig

#############################################################################################################################################################################################


def imbalanced_features_manager(df):
    ls = imbalanced_feature(df)
    if ls == []:
        info("There are no imbalanced Features in Dataset")
    else:
        Markdown_Style("Imbalanced Features in Dataset are : ", 2)
        dataframe(ls)
    text("")
    text("")


#############################################################################################################################################################################################


def useless_features_manager(df):
    text("")
    Markdown_Style("Useless Features :", type=2)
    write("The features which have high unique values are:")
    usl_df = useless_feat(df)

    # appending the id column[if any] present in given df
    # Checking for id column
    lis_wow = []
    for feature_name in df.columns:
        if 'id' in feature_name.lower():
            lis_wow.append(feature_name)
    lis_wow.extend(list(usl_df['Feature']))
    usl_df = pd.DataFrame(lis_wow, columns=['Feature'])

    if len(usl_df) != 0:
        text("")
        lis = []
        for feature in usl_df["Feature"]:
            if checkbox('Select to drop ' + feature):
                drop_useless_feat(df, feature)
                lis.append(feature)
                success("Feature Dropped Successfully")

        new = list(usl_df['Feature'])
        for val in lis:
            new.remove(val)
        usl_df = usl_df[usl_df['Feature'].isin(
            new)].reset_index().drop('index', axis=1)

        if len(usl_df) != 0:
            write("Useless Features present in DataFrame : ")
            write(usl_df)
        text("")
        text("")
    else:
        info("There are no useless Features in dataset")


#############################################################################################################################################################################################


def final_summary_provider(df):

    markdown(
        "____________________________________________________________________________")
    header("After Doing the Data Preprocessing The dataset is now as below")
    markdown(
        "____________________________________________________________________________")

    text("")
    dataframe(df.head())

    if sum(df.isnull().sum()) == 0:
        text("")
        success(
            "Congrats Data Preprocessing phase is Done 🎉🎉. Now You can move on to next part, i.e , Doing EDA")

    # text("")
    # success("Congrats Feature Engineering is Done 🎉🎉. Now You can move to next part, i.e , Doing EDA")

    text("")
    info("To download this updated dataset click the link below")
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    markdown(href, unsafe_allow_html=True)
