from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from modules.models import Models
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'Examplar-datasets/titanic.csv')
Target_feature = 'Survived'
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())


# ==============================================================================================
'''
getting each feature and showing the ouput acc to:-
Categorical feature
>>> numerical
>>> non-numeric
Numerical feature
>>> discreate
>>> continous
'''


cat_num = []
cat_n_num = []
num_dis = []
num_cont = []

for columns in df.columns:
    col_type = df[columns].dtype        #getting feature type

    if col_type == 'int64' or col_type == 'int32':
        col_unique_len = len(df[columns].unique())      #len count for unique values of any feature
        if col_unique_len <= 3:
            print(columns, "categorial feature; numerical")
            cat_num.append(columns)
        else:
            print(columns, "Numerical feature; discreate")
            num_dis.append(columns)

    elif col_type == 'object':
        print(columns, "Categorial feature; non-numeric")
        cat_n_num.append(columns)

    elif col_type == 'float64' or col_type == 'float32':
        print(columns, "Numerical feature; continous")
        num_cont.append(columns)


# getting the columns types dictionary
col_types_dict = dict(
            CategoricalFeature = dict(
                numerical = cat_num,
                non_numerical = cat_n_num,
            ),
            NumericalFeature = dict(
                discreate = num_dis,
                continous = num_cont,
            )
        )
print(col_types_dict)
print('\n')


# ==============================================================================================
total_len = df.shape[0]
for column in df.columns:
    null_count = df.isnull().sum()[column]
    percentage = round((null_count/ total_len)*100, 2)
    
    if percentage <= 40:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        elif len(df[columns].unique()) <= 3 and (df[column].dtype == "int64" or df[column].dtype == "int32"):
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    else:
        df.drop(column, axis=1, inplace=True)
        if column in cat_n_num:
            cat_n_num.remove(column)
        elif column in cat_num:
            cat_num.remove(column)
        elif column in num_cont:
            num_cont.remove(column)
        elif column in num_dis:
            num_dis.remove(column)


# ==============================================================================================
'''
If the categorical feature has null values less or equal to 40 percent then the given below code will directly fill the null values with the mode of the particular feature
'''
## features that has only categorical with non-numeric values.
# total_len = df.shape[0]
# for columns in cat_n_num:
#     #Calculating null counts for each feature
#     null_count = df.isnull().sum()[columns]

#     # Calculating the percentage for null values
#     percentage = round((null_count/total_len)*100, 2)

#     # If the percentage of null values is less then 40 percent then fill the Nan places with the mode of the particular feature
#     if percentage <= 40:
#         df[columns] = df[columns].fillna(df[columns].mode()[0])
#         print('-'*50)
#         print(df[[columns, Target_feature]].groupby([columns], as_index=False).mean())


# ## features that has only categorical with numeric values.
# total_len = df.shape[0]
# for columns in cat_num:
#     #Calculating null counts for each feature
#     null_count = df.isnull().sum()[columns]

#     # Calculating the percentage for null values
#     percentage = round((null_count/total_len)*100, 2)

#     # If the percentage of null values is less then 40 percent then fill the Nan places with the mode of the particular feature
#     if percentage <= 40:
#         df[columns] = df[columns].fillna(df[columns].mode()[0])
#         if columns != Target_feature:
#             print('-'*50)
#             print(df[[columns, Target_feature]].groupby([columns]).mean())


# # ===============================================================================================
# '''
# filling/droping nan values of numerical features
# '''
# ## Numerical features that has only discreate values.
# total_len = df.shape[0]
# for columns in num_dis:
#     #Calculating null counts for each feature
#     null_count = df.isnull().sum()[columns]

#     # Calculating the percentage for null values
#     percentage = round((null_count/total_len)*100, 2)

#     # If the percentage of null values is less then 40 percent then fill the Nan places with the mode of the particular feature
#     if percentage <= 40:
#         df[columns] = df[columns].fillna(df[columns].mean())
#         if columns != Target_feature:
#             print('-'*50)
#             print(df[[columns, Target_feature]].groupby([columns], as_index=False).mean())


# ## Numerical features that has only continous values.
# total_len = df.shape[0]
# for columns in num_cont:
#     #Calculating null counts for each feature
#     null_count = df.isnull().sum()[columns]

#     # Calculating the percentage for null values
#     percentage = round((null_count/total_len)*100, 2)

#     # If the percentage of null values is less then 40 percent then fill the Nan places with the mode of the particular feature
#     if percentage <= 40:
#         df[columns] = df[columns].fillna(df[columns].mean())
#         if columns != Target_feature:
#             print('-'*50)
#             print(df[[columns, Target_feature]].groupby([columns], as_index=False).mean())


# # ==============================================================================================
# '''Droping the features'''
# for column in df.columns:
#     null_value = df.isnull().sum()[column]
#     if null_value != 0:
#         df.drop(column, axis=1, inplace=True)
#         if column in cat_num:
#             cat_num.remove(column)
#         elif column in cat_n_num:
#             cat_n_num.remove(column)
#         elif column in num_dis:
#             num_dis.remove(column)
#         else:
#             num_cont.remove(column)


# ==============================================================================================
'''Creating len column for object feature'''

def word_count(x):
    count = 0
    for single_word in x.split():
        count += len(single_word)
    return count

for column in df.columns:
    if df[column].dtype == "object" and len(df[column].unique()) >= 10:
        df[column + "_word_count"] = df[column].apply(lambda x: word_count(x))
        df[column + "_count"] = df[column].apply(lambda x: len(x.split()))
        df[column + 'Title'] = df[column].str.extract('([A-Za-z]+)\.', expand=False)
        try:
            df[column + "Title"] = df[column + "Title"].apply(lambda x: len(x))
        except:
            try:
                df.drop([column + "Title"], axis = 1, inplace = True)
            except:
                pass

print('\n')
print('\n')
print(df.head())
print('\n')
print('\n')


# ===============================================================================================
'''
Transforming the features label and one hot encoding
'''
print(df.isnull().sum())
for columns in cat_n_num:
    unique_len_percentage = (len(df[columns].unique())/total_len)*100
    if len(df[columns].unique()) <= 5 and columns != Target_feature:

        #Creating a unique dict for value counts
        unique_dict = dict(df[columns].value_counts())

        #Creating a total len for value counts
        len_unique_values = len(df[columns].value_counts())

        #Creating range upto len of unique values 
        range_unique = range(len_unique_values)

        #Creating list for keys of unique dictionary
        unique_list = list(unique_dict.keys())

        for uniques, keys in zip(range_unique, unique_list):
            unique_dict[keys] = uniques

        df[columns] = df[columns].map(unique_dict).astype(int)

    elif unique_len_percentage <= 20 and columns != Target_feature:
        df = pd.get_dummies(df, columns = [columns], drop_first = True)


# ==============================================================================================
for column in df.columns:
    unique_len_percentage = (len(df[column].unique())/total_len)*100
    if df[column].dtype == 'object' and unique_len_percentage > 20:
        print()
        print(column)
        df.drop(column, axis=1, inplace=True)
        if column in cat_n_num:
            cat_n_num.remove(column)


# ==============================================================================================
'''
Removing the id column if that id is of numerical plus des type and the length of the column must be comarable to the actual length of the data also the column name must contain the word "id"
'''
id_df = pd.DataFrame()
for column in num_dis:
    the_col = ''
    for words_index in range(len(column)-1):
        id_word = (column[words_index] + column[words_index + 1]).lower()
        if id_word == 'id':
            the_col += column
    if the_col:
        id_unique_len_prcntage = (len(df[the_col].unique())/total_len)*100

        if id_unique_len_prcntage >= 80:
            # to save id column for later use
            the_id = pd.DataFrame(df[column], columns = [column])
            id_df = pd.concat([id_df, the_id], axis = 1, ignore_index = True)
            id_df.rename(columns={0: column}, inplace=True)

            # removing the id column from the main dataframe
            df.drop(column, axis = 1, inplace = True)
            num_dis.remove(column)


# ==============================================================================================
'''Removing outliers'''
#find Q1, Q3, and interquartile range for each column
for column in df.columns:
    if df[column].dtype == "float64" or df[column].dtype == "float32" and df[column] != Target_feature or (df[column].dtype == "int64" and len(df[column].unique()) >= 20):
        Q1 = df[column].quantile(q=.25)
        Q3 = df[column].quantile(q=.75)
        IQR = df[column].apply(stats.iqr)
        print('Q1--->', Q1)
        print('Q3--->', Q3)
        print("IQR--->", IQR)
        Q1_value = (Q1-1.5*IQR)[0]
        Q3_value = (Q3+1.5*IQR)[0]

        # Q3_value = df[column].quantile(q=.75)

        #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
        df[column] = df[column].apply(lambda x: Q3 if x > Q3_value else x)
        df[column] = df[column].apply(lambda x: Q1 if x < Q1_value else x)
        
        print('\n')
        # print(df_clean)
        print(Q3)
        print('\n')
        print(Q1)

        # find how many rows are left in the dataframe
        # print(df_clean.shape)


# ==============================================================================================
'''Splitting into training and validation datasets'''
X = df.drop(Target_feature, axis = 1)
y = df[Target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)   #shuffle can be false.
print(f"Shape of X_train--->{X_train.shape}, Shape of X_test---> {X_test.shape}")
print(f"Shape of y_train--->{y_train.shape}, Shape of y_test---> {y_test.shape}")

print(X.head(5))
print(X.columns)


# ============================================================================================
'''Normalization; Feature scaling'''
# # fit scaler on training data
# norm = MinMaxScaler().fit(X_train)

# # transform training data
# X_train_norm = norm.transform(X_train)

# # transform testing dataabs
# X_test_norm = norm.transform(X_test)


# ==============================================================================================
'''Standardization; Feature scaling'''
# # copy of datasets
# X_train_stand = X_train.copy()
# X_test_stand = X_test.copy()


# # apply standardization on numerical features
# for column in X.columns:
#     if X[column].dtype == "float64" or X[column].dtype == "float32" or (X[column].dtype == "int64"
#             and len(X[column].unique()) >= 4) and (column != Target_feature):
#         # fit on training data column
#         scale = StandardScaler().fit(X_train_stand[[column]])

#         # transform the training data column
#         X_train_stand[column] = scale.transform(X_train_stand[[column]])

#         # transform the testing data column
#         X_test_stand[column] = scale.transform(X_test_stand[[column]])


# ==============================================================================================
model_object = Models([X_train, X_test], [y_train, y_test],
                      "classification", model_list=['LogisticRegression'])

model_object.model_call(hypertunnig=True)
# ==============================================================================================
print('=====================================')

print('\n')
print(df.head())
print(df.isnull().sum())

# ==============================================================================================
## outliers, done, [can be improved by spreding outliers on a specific range], pending.
# normal distribution.
# name feature.
# location feature.
# date feature.
# if any feature has a such value that is prety much low then just fill it with another value based on some criteria.
# value_count feature
# dealing with date column
# normalizing/ standardize, (comparison b/w thier accuracies)
# linear combinations, (creation of artificial feature compare two features with * then check for the acc each time).
# if the length of columns are more then 50 then use memory reduction.
# df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False), (complete this also)
# hyperparameter tunning, voting technique, stacking, blending

## dtypes shown, done
## category feature with less then 40% of null values filled with mode of the feature, done.
## fill nan values for int or float type object, done.
# modularize nan filler.
## category feature with unique values are less then 6 then kind of label encoding is done, done.
## if unique values are less then 20 percent then one hot encoding is done, done.
## droping the features with more then 40 % null values, done.
## drop the id if its a int64 object and unique value prcentage is comparable to the len of dataset, done.
## how can i access the id later on, done (saved in `id_df`).
## spliting the dataset into training and test data, done, [if date column is present then don't do random split], pending.
# label encoding for target feature if it is of object type.
# classify the problem statement(regression/ classification).
# track imbalanced features.
# if unique value for object feature is more then i have to add another feature with the len of the elements of that feature.


# >>> filling null values, drop acc to null values.
# >>> transformation, label and one hot encoding
# >>> feature scaling, fixing outliers
# >>> feature engineering, Nan.
# >>> feature selection, Nan.
# >>> model training, done.
# >>> hyperparameter optimization.
