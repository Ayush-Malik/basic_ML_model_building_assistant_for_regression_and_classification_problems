# import sys
# sys.path.append(
#     r'E:\_ml_automator\basic_ML_model_building_assistant_for_regression_and_classification_problems')

from pandas.core.arrays.sparse import dtype

from automation import DataCleaner  # Must specify the correct path use init.py file.
import pandas as pd

# -------------------------------------
#               Testing
# -------------------------------------

# ---------- only one data ------------
df = pd.read_csv(
    r'E:\_ml_automator\basic_ML_model_building_assistant_for_regression_and_classification_problems\example_datasets\titanic.csv')

# ------- defining the object ---------
df_obj = DataCleaner(df)

# ---------- column names -------------
def test_column_name():
    col_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
        'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    assert df_obj.column_name == col_names

# --------- tracking column -----------
def test_tracking_col():
    col_name_dtype = {'PassengerId': dtype('int64'), 'Survived': dtype('int64'), 'Pclass': dtype('int64'), 'Name': dtype('O'), 'Sex': dtype('O'), 'Age': dtype(
        'float64'), 'SibSp': dtype('int64'), 'Parch': dtype('int64'), 'Ticket': dtype('O'), 'Fare': dtype('float64'), 'Cabin': dtype('O'), 'Embarked': dtype('O')}

    assert df_obj.track_col() == col_name_dtype

# ----- only categorical column -------
def test_cat_col():
    cat_col = ['Survived', 'Pclass', 'Name',
               'Sex', 'Ticket', 'Cabin', 'Embarked']
    assert df_obj.cat_col() == cat_col

# ------ only numerical column --------
def test_num_col():
    num_col = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']
    assert df_obj.num_col() == num_col

# --------- is cat column? ------------
def test_is_cat():
    val = "Only('Name',) is of categorical type."
    assert df_obj.is_cat(['Name', 'Age']) == val

# --------- is num column? ------------
def test_is_num():
    val = " Only ('Age',) is of numeric type."
    assert df_obj.is_num(["Name", "Age"]) == val

# ------- head of the dataset ---------
def test_head():
    assert df_obj.head() == df.head()

# ------- shape of the dataset --------
def test_shape():
    assert df_obj.shape == df.shape

# --------- checking isnull -----------
def test_isnull():
    assert df_obj.isnull() == df.isnull()

# ----------- isnull sum --------------
def test_isnull_sum():
    assert df_obj.isnull_sum() == df.isnull().sum()

# -------- manual data cleaner --------
def test_data_cleaner():
    df_obj.manual_process('Cabin', 'drop')
    df.drop('Cabin', axis=1, inplace=True)
    assert df_obj.column_name == df.columns()
