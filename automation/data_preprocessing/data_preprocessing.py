from automation import DataType
import pandas as pd

__all__ = [
    "FeatTransform",
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
                iqr = self.inter_quartile_range(columns)
                Q1_value = (Q1 - 1.5*iqr)[0]
                Q3_value = (Q3 + 1.5*iqr)[0]
                
                self.data[columns] = self.data[columns].apply(
                    lambda x: Q3 if x > Q3_value else x)
                
                self.data[columns] = self.data[columns].apply(
                    lambda x: Q1 if x < Q1_value else x)
