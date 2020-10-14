'''
    This module is used to perform some basic but also neccessary tasks upto null value counts.
    Some of these are:-
        " Finding the **datatypes** for each feature "
        " Getting the **columns' name** and keeping track on them "
        " Finding the **null** values in the feature and giving the appropriate strategy 
        to deal with them "
'''
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import scipy.stats as stats

__all__ = [
    "Basic",
    "Columns",
    "DataType",
    "DataCleaner",
]

class Basic:
    '''This is the base class of ml-automator that serves various impartant basic task 
    required in the automation for any ml use-case.
    
    Functionalities
    =======
    1. concating the dataframe/ series; same as concat of pandas.
    2. display head of the dataset/ dataframe.
    3. display shape of the dataset/ dataframe.
    4. getting the unique values of any particular column of the dataset passed as
    a parameter.
    5. getting the percentage of unique values of a particular column of the dataset.
    6. getting the value counts of a particular column of the dataset.
    7. map the values in dict object into the column parameter passed.
    8. also has get_dummies(same as pandas).
    9. getting isnull values of the dataset/ dataframe.
    10. getting sum of isnull values of the dataset/dataframe.
    11. getting the total length of the dataset.
    12. getting isnull sum percentage for any particular column.
    13. any particular column can also be drop by using `drop` method.
    14. also has capability of creating a dataframe.
    15. central tendency finder is also here that can be used to find (mean, median, or mode)
    for any particular column.
    16. inter quartile range for any numeric column present in the dataset.
    17. first quantile for any numeric column present in the dataset.
    18. third quantile for any numeric column present in the dataset.
    19. minimum value of any column.
    '''
    def __init__(self):
        raise NotImplementedError("Not accessible directly")

    def concat(self, *args, **kwargs):
        ''' arg is used to take dataframes/ series type objects(one or more then one)
        and kwargs is used to take parameters required to concate two dataframes/ series object. '''
        self.data = pd.concat(objs=args, **kwargs)
        return self.data

    def head(self, value=5):
        ''' Returns head of the dataset. '''
        return self.data.head(value)

    @property
    def shape(self):
        ''' Returns the shape of the dataset. '''
        return self.data.shape

    def unique(self, column_name):
        return self.get_col(column_name).unique()

    def unique_prcntg(self, column_name):
        return round((len(self.unique(column_name)) / self.data_len)*100, 2)

    def value_counts(self, column_name):
        return self.get_col(column_name).value_counts()

    def map(self, column_name, dict_obj):
        return self.get_col(column_name).map(dict_obj)

    def get_dummies(self, column_name):
        ''' limited version of pd.get_dummies... Only creates dummy variables for the column
        parameter passed. '''
        return pd.get_dummies(self.data, columns=[column_name], drop_first=True)

    def isnull(self) -> DataFrame:
        ''' return the isnull value from dataframe '''
        return self.data.isnull()

    def isnull_sum(self, column_name=None):
        ''' return sum of null values present in the dataset. '''
        if column_name:
            return self.isnull().sum()[column_name]
        return self.isnull().sum()

    @property
    def data_len(self):
        ''' return the total length of the dataset'''
        return self.shape[0]

    def isnull_sum_prcntg(self, column_name):
        ''' Return the percentage of sum of null values present in the dataset. '''
        if column_name:
            return round((self.isnull_sum(column_name) / self.data_len)*100, 2)
        return round((self.isnull_sum / self.data_len)*100, 2)

    def drop(self, column, **kwargs):
        ''' Can be used to drop any particular feature from the dataset. '''
        if not isinstance(column, list):
            column = [column]
        return self.data.drop(column, **kwargs)

    @staticmethod
    def dataframe(*args, **kwargs):
        ''' Used to create a dataframe same as pd.DataFrame '''
        return pd.DataFrame(*args, **kwargs)

    def central_tendency_finder(self, column_name, measure):
        ''' return the measure(mean, median, mode) of selected column in the dataset
        according to the parameter(measure) is passed. '''
        measure = measure.lower()
        column = self.get_col(column_name)
        if measure == "mode":
            return column.mode()[0]
        elif measure == "mean":
            return column.mean()
        elif measure == "median":
            return column.median()
        else:
            raise AttributeError(f"Wrong attribute is passed; {measure} measure is not recognized")

    def inter_quartile_range(self, column_name):
        """
        Returns the inter quartile range for any particular feature.
        The inter quartile range is the difference b/w Q3 and Q1 i.e.
        75% - 25% and basically its the midspread of the data.
        """
        return self.get_col(column_name).apply(stats.iqr)

    def Q1(self, column_name):
        """
        Returns the first quantile for the column of the dataset.
        """
        return self.get_col(column_name).quantile(q=.25)

    def Q3(self, column_name):
        """
        Returns the third quantile for the column of the dataset.
        """
        return self.get_col(column_name).quantile(q=.75)

    def col_minval(self, column_name):
        """
        Returns the min value of the feature
        """
        return self.get_col(column_name).min()


class Columns(Basic):
    '''Takes in the main data i.e. dataframe object performs certain task to deal
    with features of the dataset.
    
    Example
    ========
    >>> df = Column(dataframe)
    >>> df.column_name
    return "total column names as list type object."
    
    >>> df.cat_col()
    return "All categorical features of the dataset if present; return type list"
    
    >>> df.num_col()
    return "All numerical features of the dataset if present; return type list"
    
    >>> df.is_cat([column])
    return True "if column is of categorical type"
    
    >>> df.is_cat([columns,])
    return True "if all columns are of categorical type"
    if "all columns are not of categorical type":
        return "column name that are of categorical type"
    else:
        return "column is not of categorical type"
    
    >>> df.is_num([column/ columns])     
    # Does the same work as done by is_cat but this time its for numerical feature
    
    >>> df.track_col()    # ------- this method is used by child class to overwrite.
    '''

    def __init__(self, *args):
        ''' Used to take dataframe objects. '''
        if len(args) == 1:
            self.data = args[0]
        else:
            # Data can be already divided into train and test datas.
            self.data = self.concat(*args, **dict(axis=0, ignore_index=True))
        # original data then later on be accessed by this.
        self.get_data = self.data.copy()

    @property
    def column_name(self):
        ''' returns name of all columns present in the dataset. '''
        return list(self.data.columns)

    def get_col(self, column_name):
        ''' returns the values of specified column. '''
        return self.data[column_name]

    def col_len(self):
        ''' return the total length of columns of the dataset '''
        return self.shape[1]

    def cat_col(self):
        ''' Keeps track of only categorical features of the dataset '''
        _dict = self.data_type()
        categorical_col = _dict.get("categorical_feature", None)
        if categorical_col is None:
            return("There is no categorical feature present in the dataset.")
        return categorical_col

    def num_col(self):
        ''' Keeps tracks of only numerical features of the dataset '''
        _dict = self.data_type()
        numerical_col = _dict.get("numerical_feature", None)
        if numerical_col is None:
            return("There is no numerical feature present in the dataset.")
        return numerical_col

    def is_cat(self, *args):
        '''args takes column names as list type object'''
        _dict = self.data_type(*args)
        categorical_col = _dict.get("categorical_feature", None)
        
        if categorical_col is None:
            return(f"{args[0]} not found.")
        
        if categorical_col == args[0]:
            return True
        return f" Only {tuple(categorical_col)} is of categorical type."

    def is_num(self, *args):
        '''args takes column names as a list type object'''
        _dict = self.data_type(*args)
        numerical_col = _dict.get("numerical_feature", None)
        if numerical_col is None:
            return(f"{args[0]} not found")
        
        if numerical_col == args[0]:
            return True
        
        return f" Only {tuple(numerical_col)} is of numeric type."

    def track_col(self):
        ''' Track all of the columns along with their datatypes;
        implemented in the child class. '''
        pass


class DataType(Columns):
    ''' Finds datatypes of every features and categorize the features accordingly into
    categorical or numerical features. '''
    def __init__(self, *args) -> None:
        Columns.__init__(self, *args)

    def track_col(self, *args):
        ''' Args takes column name as list type object.
        
        This method is used to get datatypes of features and return the output as
        dictionary object.
        >>> return {column_name: column_type}
        '''
        column_name = None
        if args:
            column_name = args[0]
        else:
            column_name = self.column_name
        df = self.data
        self._dict = dict((features, df[features].dtype) for features in column_name)
        return self._dict

    def data_type(self, *args):
        ''' Args take column names as list type object.
        >>> data_type([columns])
        
        If column names are passed all of
        the computation is done for the particular columns but if not the default computation
        is done for each feature of the dataset.
        
        Return a dictionary object with names of categorical and numerical features.
        
        >>> return {'categorical_feature': [column_names], 'numerical_feature': [column_names]}
        '''
        features_type = None
        cat_list = []
        num_list = []
        
        if args:
            features_type = self.track_col(*args).items()
        else:
            features_type = self.track_col().items()
        
        for col_name, col_type in features_type:
            if 'int' in str(col_type):
                if self.unique_prcntg(col_name) <= 0.5:
                    # categorial feature; numerical type
                    cat_list.append(col_name)
                else:
                    # Numerical feature; discreate type
                    num_list.append(col_name)
                    
            elif col_type == "object":
                # Categorial feature; non-numeric type
                cat_list.append(col_name)
            elif 'float' in str(col_type):
                # Numerical feature; continous type
                num_list.append(col_name)
            else:
                raise AttributeError(f"Wrong datatype encountered; {col_type} is not recognized.")
            
        return dict(
            categorical_feature = cat_list,
            numerical_feature = num_list,
        )


class DataCleaner(DataType):
    def auto_process(self):
        ''' This method automatically finds the best way for getting rid of columns
        having null values in them. Either this method would delete the particular column or
        just fill that with appropriate value. '''
        done = None
        cntrl_tndncy_meas = None
        for column in self.column_name:
            null_percentage = self.isnull_sum_prcntg(column)
            if null_percentage == 0:
                done = 'skipped'
            elif  null_percentage <= 40:
                if self.is_cat([column]):
                    cntrl_tndncy_meas = self.central_tendency_finder(column, 'mode')
                else:
                    cntrl_tndncy_meas = self.central_tendency_finder(column, 'mean')
                done = 'filled'
                self.null_filler(column, cntrl_tndncy_meas)
            else:
                done = 'dropped'
                self.drop(column, axis=1, inplace=True)
            
            if done == "skipped":
                yield(f"The {column} column is {done} as it didn't have any kind of null values")
            elif done == "dropped":
                yield(f"The {column} column is {done} as it has {null_percentage}% of the null values.")
            else:
                yield(f"The {column} column is {done} with the value of {cntrl_tndncy_meas}")

    def manual_process(self, column_name, parameter) -> None:
        """
        This takes a parameter from the user and according to that it process the data.
        
        parameter:- input type must be of str. It takes one of the (Drop, mean, median, mode)
        value.
        """
        if "drop" in parameter:
            self.drop(column_name, axis=1, inplace=True)
        else:
            value = self.central_tendency_finder(column_name, parameter)
            self.null_filler(column_name, value)

    def id_remover(self):
        ''' This method is used to automatically remove id column from the dataframe.
        It also returns the id column. '''
        for column in self.num_col():
            is_id = False
            
            if "id" in column.lower():
                is_id = True
            
            if is_id:
                prcntage = self.unique_prcntg(column)
                if prcntage >= 80:
                    self.id_col = self.dataframe(self.get_col(column), columns=[column])
                    self.drop(column=column, axis=1, inplace=True)
                    return self.id_col
                return f"Is the {column} column is the ID for the dataset."
        return "There is no Id column present in the dataset."

    def null_filler(self, column_name, value) -> None:
        ''' Fill the null values of the feature with the attribute(value) passed '''
        self.data[column_name] = self.get_col(column_name).fillna(value)


# must save the original data
# concat option---> done..
# column_name option ---> done..
# datatype_tracker ---> done..
