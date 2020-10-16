from sklearn.utils import shuffle
from automation import DataType
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "FeatTransform",
    "Outliers"
]
# track target feature also.
class FeatTransform(DataType):
    def auto_encoder(self):
        for column in self.cat_col():
            percentage = self.unique_prcntg(column)
            if len(self.unique(column)) <= 5:
                self.label_encoder(column)
            
            elif percentage <= 20:
                self.one_hot_encoder(column)
            
            else:
                self.drop(column, axis=1, inplace=True)

    def label_encoder(self, column):
        #Creating a unique dict for value counts
        unique_dict = dict(self.value_counts(column))
        
        #Creating a total len for value counts
        len_unique = len(self.value_counts(column))
        
        #Creating range upto len of unique values
        range_unique = range(len_unique)
        
        #Creating list for keys of unique dictionary
        unique_list = list(unique_dict.keys())
        
        for uniques, keys in zip(range_unique, unique_list):
            unique_dict[keys] = uniques
        
        self.data[column] = self.map(column, unique_dict).astype(int)

    def one_hot_encoder(self, column):
        self.data = self.get_dummies(column)


class Outliers(DataType):
    def outliers(self):
        for columns in self.num_col():
            if self.unique_prcntg(columns) >= 20:
                Q1 = self.Q1(columns)
                Q3 = self.Q3(columns)
                
                self.fix_outliers(
                    value_1=Q1,
                    value_3=Q3,
                    threshold=1.5,
                    column_name=columns
                )

    def fix_outliers(self, value_1, value_3, threshold, column_name):
        if not isinstance(threshold, float):
            try:
                threshold = float(threshold)
            except:
                raise AttributeError("The type of threshold in not valid.")
        
        if self.is_num([column_name]):
            Q1 = self.Q1(column_name)
            Q3 = self.Q3(column_name)
            iqr = self.inter_quartile_range(column_name)
            lower_bound = (Q1 - threshold*iqr)[0]
            upper_bound = (Q3 + threshold*iqr)[0]
            
            self.data[column_name] = self.get_col(column_name).apply(
                lambda x: value_1 if x < lower_bound else x)
            self.data[column_name] = self.get_col(column_name).apply(
                lambda x: value_3 if x > upper_bound else x)
        else:
            raise AttributeError("The column is not of numeric type. Must pass column only of numeric type.")


class DataSplitter(DataType):
    def data_splitter(self, test_size=0.33):
        if self.data_num > 1:
            train_data_len = self.data_len_list[0]
            train_data = self.data.iloc[: train_data_len, :]
            test_data = self.data.iloc[train_data_len: , :]
            
            X_train= train_data.drop(self.target_feat, axis=1)
            X_test = test_data.drop(self.target_feat, axis=1)
            y_train = train_data[self.target_feat]
            y_test = test_data[self.target_feat]
            return (X_train, X_test, y_train, y_test)
        
        return self.train_test_splitter(test_size=test_size)

    def train_test_splitter(self, test_size):
        shuffle = True
        if self.is_date_type:
            shuffle = False
        
        X = self.drop(self.target_feat, axis=1)
        y = self.get_col(self.target_feat)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=42)
        
        return (X_train, X_test, y_train, y_test)


class ImbalancedDataset(DataType):
    pass
