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
