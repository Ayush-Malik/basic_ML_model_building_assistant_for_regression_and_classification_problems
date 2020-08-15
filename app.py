from streamlit import *
import pandas as pd 
import numpy as np 
from feature_eng import *
from EDA import *
from models import *
import base64


activities = ["Home", "EDA", "Model Building", "About Us"]	
choice = sidebar.selectbox("Select Option",activities)
set_option('deprecation.showfileUploaderEncoding', False)
markdown_style = "position: relative; left: 50px; font-size:30px; color:grey; font-family: Brush Script MT;"
markdown_style2 = "position: relative; font-size:30px; color:brown; font-family: Algerian;"
markdown_head = "text-align: center; font-family: Georgia, Times, serif; font-weight: bolder; font-size:40px; padding-top: 20px; background-image: linear-gradient(to left, rgb(184, 48, 184), rgb(59, 9, 95), blue); - webkit-background-clip: text; - moz-background-clip: text; background-clip: text; color: transparent; "



if choice == "Home": # For Navigating to Home Page
    markdown("<h1 style='"+ markdown_head +"'>Feature Engineering</h1 >",
            unsafe_allow_html=True)
    text("")
    text("")

    markdown("<p style='" + markdown_style2 +
             "' >Upload the dataset!!!</p>", unsafe_allow_html=True)
    data = file_uploader("" , type=["csv"]) # Loading the dataset


    if data != None : # Here if block runs only when user gives dataset
        df = pd.read_csv(data)
        
        text("")
        text("")  

        markdown("<p style='" + markdown_style2 +
                 "' >Head of the Dataset:- </p>", unsafe_allow_html=True)
        text("")

        dataframe(df.head())
        text("")

        markdown("<p style='" + markdown_style +
                 "' >Shape of the Dataset:-  "+ str(df.shape) + "</p>", unsafe_allow_html=True)
        text("")
        text("")

        dic = type_of_feature(df)
        markdown("<p style='" + markdown_style2 +
                 "' >Categories of Features:- </p>", unsafe_allow_html=True)
        write(dic)
        text("")

        missing_values_count = null_value(df)
        markdown("<p style='" + markdown_style2 +
                 "' >The Missing Values and Strategey:- </p>", unsafe_allow_html=True)
        write(missing_values_count)
        text("")
        text("")

        markdown("<p style='" + markdown_style +
                 "' >Heatmap for null values</p>", unsafe_allow_html=True)
        write("")
        text("")
    
        heat_plot = heatmap_generator(df.isnull())
        pyplot()
        text("")
        text("")

        ls = imbalanced_feature(df)

        if ls == []:
            info("There are no imbalanced Features in Dataset")
        else:
            markdown("<p style='" + markdown_style2 +
                     "' >Imbalanced Features in Dataset are:- </p>", unsafe_allow_html=True)
            dataframe(ls)
        text("")
        text("")

        categorical = cat_num(df)

        if checkbox("Show value count of a Categorical feature"):
            new_cat = ["Choose The Feature"]
            new_cat.extend(categorical)
            categorical_feature = selectbox("", new_cat)
            if categorical_feature != "Choose The Feature":
                unique_len = len(df[categorical_feature].value_counts())
                if unique_len > 15:
                    dataframe(df[categorical_feature].value_counts())
                    markdown("<p style='" + markdown_style +
                             "' >Total unique values:-" + str(unique_len) + "</p>", unsafe_allow_html=True)
                else:
                    percent_pie = prcntage_values( categorical_feature, df)
                    plotly_chart(percent_pie)
            
        text("")
        if checkbox("Show comparison b/w two categorical features"):
            markdown("<p style='" + markdown_style + "' >Two categorical features values combined comparator</p>", unsafe_allow_html=True)

            new_cat = ["Select First Categorical Feature"]
            new_cat.extend(categorical)
            categorical1 = selectbox("", new_cat)
            
            if categorical1 != "Select First Categorical Feature":
                new_cat2 = ["Select Second Categorical Feature"]
                new_cat2.extend(categorical)
                new_cat2.remove(categorical1)
                categorical2 = selectbox("", new_cat2)
            
            if categorical1 != "Select First Categorical Feature" and categorical2 != "Select Second Categorical Feature":
                cat_lis = [categorical1, categorical2] 
                comparison_plot = two_cat_comparator( cat_lis , df )
                try:
                    plotly_chart(comparison_plot)
                except:
                    dataframe(comparison_plot)
                    markdown("<p style='" + markdown_style +
                             "' >x-axis--> "+ categorical1 + ", y-axis--> " + categorical2 +"</p>", unsafe_allow_html=True)

        text("")
        markdown("<p style='" + markdown_style2 +
                 "' >Select the feature to be dropped:-</p>", unsafe_allow_html=True)
        missing_lis = missing_value_lis(df)
        lis_drop = multiselect("", missing_lis) 
        feature_tracker, sent = drop_feat(df, lis_drop)
        if lis_drop != []:
            success(sent)
        
        text("")
        markdown("<p style='" + markdown_style2 +
                 "' >Select Features to be filled:-</p>", unsafe_allow_html=True)
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

        text("")
        markdown("<p style='" + markdown_style2 +
                 "' >Useless Features:-</p>", unsafe_allow_html=True)
        write("The features which have high unique values are:")
        usl_df = useless_feat(df)
        write(usl_df)
        text("")
        for feature in usl_df["Feature"]:
            if checkbox("Select to drop " + feature):
                drop_useless_feat(df, feature)
                success("Feature Dropped Successfully")

        text("")
        text("")
        
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
            

elif choice == "EDA": # For Navigating to Home Page
    markdown("<h1 style='" + markdown_head + "'>Exploratory data analysis</h1 >",
             unsafe_allow_html=True)
    text("")
    text("")
    df = pd.read_csv('update.csv')
    dataframe(df.head())
    text("")
    numerical_feat,categorical_features = num_num(df)
    if checkbox("Select to Visulaize Correlation heatmap"):
        selected_features = multiselect("Select Feature", numerical_feat)
        if selected_features != []:
            fig = correlation_heatmap(df , selected_features)
            plotly_chart(fig)
    
    text("")
    if checkbox("Select to Visulaize Box Plot"):  #must show an error that you cannot pass more than 2 values
        selected_features = multiselect("Select minimum two Feature", numerical_feat,key=2)
        if len(selected_features) >= 2:
            fig2 = box_plot(df , selected_features)
            plotly_chart(fig2)

    tot_lis = numerical_feat.copy()
    tot_lis.extend(categorical_features)

    text("")
    if checkbox("Select to Visualize Histo Gram"):
        selected_features = multiselect("Select Feature", tot_lis, key=4)
        if selected_features != [] and len(selected_features) <= 2:
            fig3 = histo_gram(df , selected_features)
            plotly_chart(fig3)
        elif len(selected_features) > 2:
            warning("You are trying to select excessive Features")

    new_num = numerical_feat.copy()
    newl = ["Feature"]
    newl.extend(new_num)

    text("")
    if checkbox("Select to Visualize Sun Burst Plot"):
        selected_features = multiselect("Select Feature", tot_lis, key=5)
        text("")
        vals = selectbox("Select Feature", newl)
        if selected_features != [] and vals != "Feature":
            fig4 = sun_burst(df , selected_features, vals)
            plotly_chart(fig4)

elif choice == "Model Building": # For Navigating to Home Page
    markdown("<h1 style='" + markdown_head + "'>Model Building And Training</h1 >",
             unsafe_allow_html=True)
    text("")
    text("")

    df = pd.read_csv('update.csv')
    dataframe(df.head())
    fe_list = feature_list(df)
    text("")

    text("")
    df = pd.get_dummies(df , drop_first = True)
    info("After converting Categorical Features into Numerical Ones, The current dataset is")
    dataframe(df.head())
    text("")

    text("")
    nlist = ["Feature"]
    nlist.extend(fe_list)
    markdown("<p style='" + markdown_style2 +
             "' >Select The Target Feature:-</p>", unsafe_allow_html=True) #Correlation can also be used
    # subheader(" For the Dataset")
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
    if target_feature != "Feature":
        text("")
        markdown("<p style='" + markdown_style2 +
                "' >Let's Start Splitting The Dataset</p>", unsafe_allow_html=True)
        text("")
        prcntage = 0.82
        info("By default Percentage of Training Data is 82 %")
        text("")
        if checkbox("Select to Change Training Dataset Size"):
            prcntage = slider('Select percentage',60, 86)
            write("You selected : ", prcntage, "percent for training Dataset")
            prcntage = prcntage/100
        train, test = train_test_splitter(df, prcntage)
        text("")
        
        markdown("<p style='" + markdown_style +
                    "' >Shape of the Training Dataset:-  "+ str(train.shape) + "</p>", unsafe_allow_html=True)
        text("")
        markdown("<p style='" + markdown_style +
                    "' >Shape of the Test Dataset:-  "+ str(test.shape) + "</p>", unsafe_allow_html=True)
        text("")
        text("")

        x_train, x_test, y_train, y_test = x_y_maker(target_feature, train, test)
        info("Now the Train and test dataset are splitted into x_train, x_test, y_train, y_test")

        text("")
        text("")
        x_list = [x_train, x_test]
        y_list = [y_train, y_test]
        mlists = []
        if typ == "Regression":
            mlists = ['LinearRegression','RandomForestRegressor','AdaBoostRegressor','SVR','MLPRegressor','DecisionTreeRegressor','XGBRegressor']
        else:
            mlists = ['LogisticRegression','RandomForestClassifier','AdaBoostClassifier','SVC','MLPClassifier','DecisionTreeClassifier','XGBClassifier']
        models_lists = multiselect("Select Models", mlists)
        model_object = Models(x_list, y_list, typ,models_lists)
        model_object.model_call()
        extra = ["Select"]
        extra.extend(models_lists)
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

# null values, -ve msle (model training)
# working on it show tab krna h jab user kisi model ko select kre
