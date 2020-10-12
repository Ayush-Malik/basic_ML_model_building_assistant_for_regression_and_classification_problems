import unittest
from unittest.case import TestCase
from pandas.core.arrays.sparse import dtype

from automation import DataCleaner
import pandas as pd

# -------------------------------------
#               Testing
# -------------------------------------

df = pd.read_csv(
    r'C:\Users\user\Desktop\auto\example_datasets\titanic.csv')

# ------- defining the object ---------
df_obj = DataCleaner(df)

class TestCases(unittest.TestCase):
    # ---------- column names -------------
    def test_column_name(self):
        col_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        self.assertAlmostEqual(df_obj.column_name, col_names)

    # --------- tracking column -----------
    def test_tracking_col(self):
        self.assertTrue(df_obj.track_col())

    # ----- only categorical column -------
    def test_cat_col(self):
        cat_col = ['Survived', 'Pclass', 'Name',
                'Sex', 'Ticket', 'Cabin', 'Embarked']
        self.assertAlmostEqual(df_obj.cat_col(), cat_col)

    # ------ only numerical column --------
    def test_num_col(self):
        num_col = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']
        self.assertAlmostEqual(df_obj.num_col(), num_col)

    # --------- is cat column? ------------
    def test_is_cat(self):
        val = " Only ('Name',) is of categorical type."
        assert df_obj.is_cat(['Name', 'Age']) == val

    # --------- is num column? ------------
    def test_is_num(self):
        val = " Only ('Age',) is of numeric type."
        self.assertAlmostEqual(df_obj.is_num(["Name", "Age"]), val)

    # ------- head of the dataset ---------
    def test_head(self):
        self.assertAlmostEqual(list(df_obj.head()), list(df.head()))

    # ------- shape of the dataset --------
    def test_shape(self):
        self.assertAlmostEqual(df_obj.shape, df.shape)

    # --------- checking isnull -----------
    def test_isnull(self):
        self.assertAlmostEqual(list(df_obj.isnull()), list(df.isnull()))

    # ----------- isnull sum --------------
    def test_isnull_sum(self):
        self.assertAlmostEqual(list(df_obj.isnull_sum()), list(df.isnull().sum()))

    # -------- manual data cleaner --------
    def test_data_cleaner(self):
        df_obj.manual_process('Cabin', 'drop')
        self.assertAlmostEqual(df_obj.column_name, list(df.columns))

    # ----------- value count -------------
    def test_value_count(self):
        obj_vc = df_obj.value_counts('Embarked')
        df_vc = df['Embarked'].value_counts()
        self.assertDictEqual(dict(obj_vc), dict(df_vc))

    # -------- central tendency -----------
    def test_central_tendency(self):
        mode_obj = df_obj.central_tendency_finder('Embarked', 'mode')
        mode_df = df['Embarked'].mode()[0]
        mean_obj = df_obj.central_tendency_finder('Age', 'mean')
        mean_df = df['Age'].mean()
        self.assertAlmostEqual(mode_obj, mode_df)
        self.assertAlmostEqual(mean_obj, mean_df)

    # --------------- Q1 ------------------
    def test_Q1(self):
        obj_q1 = df_obj.Q1('Age')
        df_q1 = df['Age'].quantile(q=.25)
        self.assertAlmostEqual(obj_q1, df_q1)

    # --------------- Q1 ------------------
    def test_Q3(self):
        obj_q3 = df_obj.Q3('Age')
        df_q3 = df['Age'].quantile(q=.75)
        self.assertAlmostEqual(obj_q3, df_q3)

    # ----------- get column --------------
    def test_get_col(self):
        obj_col_value = df_obj.get_col('Embarked')
        df_col_value = df['Embarked']
        self.assertAlmostEqual(list(obj_col_value), list(df_col_value))

    # --------- column min val ------------
    def test_col_minval(self):
        obj_minval = df_obj.col_minval('Age')
        df_minval = df['Age'].min()
        self.assertAlmostEqual(obj_minval, df_minval)

    # ------- more then one dataset -------
    def test_twodatas(self):
        data = {'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18]}
        data2 = {'Name': ['galla', 'uba', 'lilu', 'lala'], 'Age': [34, 24, 1, 69]}
        obj = DataCleaner(*[pd.DataFrame(data), pd.DataFrame(data2)])
        self.assertTrue(obj)
