import unittest
from automation import DataCleaner, FeatTransform
import pandas as pd

# -------------------------------------
#               Testing
# -------------------------------------

df = pd.read_csv(
    r'..\..\example_datasets\titanic.csv')

obj1 = DataCleaner(df)

for i in obj1.auto_process():
    pass
df1 = obj1.data


class TestFeatTransform(unittest.TestCase):
    def test_one_hot_encoder(self):
        df_c = df1.copy()
        df_obj = FeatTransform(df_c)
        self.assertIsNone(df_obj.one_hot_encoder('Embarked'))

    def test_label_encoder(self):
        df_c = df1.copy()
        df_obj = FeatTransform(df_c)
        self.assertIsNone(df_obj.label_encoder('Sex'))

    def test_auto_encoder(self):
        df_c = df1.copy()
        df_obj = FeatTransform(df_c)
        self.assertIsNone(df_obj.auto_encoder())
