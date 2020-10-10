from automation import Basic, Columns, DataType, DataCleaner

# track target feature also.
class FeatTransform(Columns):
    def auto_encoder(self):
        for column in self.cat_col():
            percentage = self.unique_prnctg(column)
            if len(self.unique(column)) <= 5:
                self.label_encoder(column)
            
            elif percentage <= 20:
                self.one_hot_encoder(column)
            
            else:
                self.drop(column, axis=1, inplace=True)

    def label_encoder(self, column):
        #Creating a unique dict for value counts
        unique_dict = dict(self.value_count(column))
        
        #Creating a total len for value counts
        len_unique = len(self.value_count(column))
        
        #Creating range upto len of unique values
        range_unique = range(len_unique)
        
        #Creating list for keys of unique dictionary
        unique_list = list(unique_dict.keys())
        
        for uniques, keys in zip(range_unique, unique_list):
            unique_dict[keys] = uniques
        
        self.get(column) = self.map(column, unique_dict).astype(int)

    def one_hot_encoder(self, column):
        self.data() = self.get_dummies(column)
