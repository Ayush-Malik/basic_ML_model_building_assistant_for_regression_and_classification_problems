# Data Preprocessing

> This file is only for example purpose where this shows the usage of different classes and methods avalaible in automation package. 1st phase only consists of data preprocessing phase.

Importing libraries.

```python
>>> from automation import DataCleaner, FeatTransform, Outliers
>>> import pandas as pd
```

Loading the dataset and creating the object of `DataCleaner`.

**Note:- The DataCleaner can also take more then one dataset in it for processing and all of the methods are applicable in that case also.**

```python
>>> df = pd.read_csv(r'data\titanic.csv')
>>> df_obj = DataCleaner(df.copy())
```

Getting the head of the dataset. Showing usage of `head` method.

```python
>>> df_obj.head()
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
```

Getting the shape of the dataset object by using `shape` method.

```python
>>> df_obj.shape
... (891, 12)
```

Getting the unique values for any particular feature present in the dataset by using `unique` method.

```python
>>> df_obj.unique(column_name = 'Sex')
... ['male' 'female']
```

Getting the percentage of unique values count of any feature in the dataset by using `unique_prcntg` method.

```python
>>> df_obj.unique_prcntg(column_name = 'Sex')
... 0.22
```

Getting value counts for a particular column of the dataset by using `value_counts` method.

```python
>>> df_obj.value_counts(column_name = 'Sex')
male      577
female    314
Name: Sex, dtype: int64
```

Mapping the values of any particular column according to the dict object is passed and all of this done by `map` method.

**NOTE:- It not map the values in actual dataframe instead it only returns the column with mapped values.**

```python
>>> new_col = df_obj.map(column_name = 'Sex', dict_obj = {'male': 1, 'female': 0})
>>> new_col
0      1
1      0
2      0
3      0
4      1
      ..
886    1
887    0
888    0
889    1
890    1
Name: Sex, Length: 891, dtype: int64

>>> df['Sex'] = new_col      # This would set the new column into actual dataset.
>>> df_obj.data['Sex'] = new_col    # This would set the new column into the object's dataset of DataCleaner.
```

Getting the dummies using `get_dummies`.

**NOTE:- It also does not induce any kind of impact on the main dataframe.**

```python
>>> df_obj.get_dummies(column_name = 'Ticket')
     PassengerId  Survived  Pclass                                               Name  ... Ticket_W./C. 6609  Ticket_W.E.P. 5734  Ticket_W/C 14208  Ticket_WE/P 5735
0              1         0       3                            Braund, Mr. Owen Harris  ...                 0                   0                 0                 0  
1              2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  ...                 0                   0                 0                 0  
2              3         1       3                             Heikkinen, Miss. Laina  ...                 0                   0                 0                 0  
3              4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  ...                 0                   0                 0                 0  
4              5         0       3                           Allen, Mr. William Henry  ...                 0                   0                 0                 0  
..           ...       ...     ...                                                ...  ...               ...                 ...               ...               ...  
886          887         0       2                              Montvila, Rev. Juozas  ...                 0                   0                 0                 0  
887          888         1       1                       Graham, Miss. Margaret Edith  ...                 0                   0                 0                 0  
888          889         0       3           Johnston, Miss. Catherine Helen "Carrie"  ...                 0                   0                 0                 0  
889          890         1       1                              Behr, Mr. Karl Howell  ...                 0                   0                 0                 0  
890          891         0       3                                Dooley, Mr. Patrick  ...                 0                   0                 0                 0  

[891 rows x 691 columns]
```

Getting the null values of the dataset by using `isnull` method.

```python
>>> df_obj.isnull()
     PassengerId  Survived  Pclass   Name    Sex    Age  SibSp  Parch  Ticket   Fare  Cabin  Embarked
0          False     False   False  False  False  False  False  False   False  False   True     False
1          False     False   False  False  False  False  False  False   False  False  False     False
2          False     False   False  False  False  False  False  False   False  False   True     False
3          False     False   False  False  False  False  False  False   False  False  False     False
4          False     False   False  False  False  False  False  False   False  False   True     False
..           ...       ...     ...    ...    ...    ...    ...    ...     ...    ...    ...       ...
886        False     False   False  False  False  False  False  False   False  False   True     False
887        False     False   False  False  False  False  False  False   False  False  False     False
888        False     False   False  False  False   True  False  False   False  False   True     False
889        False     False   False  False  False  False  False  False   False  False  False     False
890        False     False   False  False  False  False  False  False   False  False   True     False

[891 rows x 12 columns]
```

Getting the sum of isnull by using `isnull_sum` method. It can also give the results according to the column_name passed(default column_name is `None`).

```python
>>> df_obj.isnull_sum()
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

>>> df_obj.isnull_sum(column_name = 'Age')
... 177
```

Getting the total percentage of null values for a particular column by using `isnull_sum_prcntg` method.

```python
>>> df_obj.isnull_sum_prcntg(column_name = 'Cabin')
... 77.1
```

Getting the total length of the dataset by using `data_len`.

```python
>>> df_obj.data_len
... 891
```

Drop any particular column from the dataset by using `drop` method.

```python
>>> df_obj.drop('Cabin', axis=1, inplace=True)      # Drop the column from the actual dataset.
>>> df_obj.head()
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833        C
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000        S
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500        S
```

Returns an appropriate value(mean, median, or mode) for any particular column according to parameter passed. This is done by using `central_tendency_finder` method.

```python
>>> df_obj.central_tendency_finder(column_name = 'Age', measure = 'mean')
... 29.69911764705882

>>> df_obj.central_tendency_finder(column_name = 'Age', measure = 'mode')
... 24.0

>>> df_obj.central_tendency_finder(column_name = 'Age', measure = 'median')
... 28.0
```

Getting the inter quartile range for any particular feature by using `inter_quartile_range` method.

```python
>>> df_obj.inter_quartile_range(column_name = 'Age')
0      0.0
1      0.0
2      0.0
3      0.0
4      0.0
      ...
886    0.0
887    0.0
888    NaN
889    0.0
890    0.0
Name: Age, Length: 891, dtype: float64
```

Getting the Q1(first quartile) of any particular column from the dataset using `Q1` method.

```python
>>> df_obj.Q1(column_name = 'Age')
... 20.125
```

Getting the Q3(third quartile) of any particular column from the dataset using `Q3` method.

```python
>>> df_obj.Q3(column_name = 'Age')
... 38.0
```

Getting the minimum value of any particular column from the dataset using `col_minval` method.

```python
>>> df_obj.col_minval(column_name = 'Age')
... 0.42
```

Getting the column names as list type object from the dataset using `column_name`.

```python
>>> df_obj.column_name
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
```

Getting the column of the dataset by passing the column name. This is done under `get_col` method.

```python
>>> df_obj.get_col(column_name = 'Sex')
0        male
1      female
2      female
3      female
4        male
        ...
886      male
887    female
888    female
889      male
890      male
Name: Sex, Length: 891, dtype: object
```

Getting the length of total features present in the dataset by using `col_len` method.

```python
>>> df_obj.col_len()
... 11
```

Getting the categorical column names as a list type object by using `cat_col` method.

```python
>>> df_obj.cat_col()
['Survived', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked']
```

Getting the numerical column names as a list type object by using `num_col` method.

```python
>>> df_obj.num_col()
['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']
```

Checking for a particular column that is it is of categorical type or not. This is done by using `is_cat` method.

```python
>>> df_obj.is_cat(['Sex'])
... True
```

Checking for a particular column that is it is of numerical type or not. This is done by using `is_num` method.

```python
>>> df_obj.is_num(['Age'])
... True
```

Getting column types along with their name by using `track_col` method. It also works perfectly fine when column names are passed and showing the result according to that case.

```python
>>> df_obj.track_col()
{'PassengerId': dtype('int64'), 'Survived': dtype('int64'), 'Pclass': dtype('int64'), 'Name': dtype('O'), 'Sex': dtype('O'), 'Age': dtype('float64'), 'SibSp': dtype('int64'), 'Parch': dtype('int64'), 'Ticket': dtype('O'), 'Fare': dtype('float64'), 'Embarked': dtype('O')}

>>> df_obj.track_col(['Age', 'Embarked'])
{'Age': dtype('float64'), 'Embarked': dtype('O')}
```

Getting the categorical_feature and numerical_feature names as dict type object. This is done by `data_type` method. Same is here if multiple column names are passed directly then it'll give the result according to that.

```python
>>> df_obj.data_type()
{'categorical_feature': ['Survived', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], 'numerical_feature': ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']}

>>> df_obj.data_type(['Age', 'Embarked'])
{'categorical_feature': ['Embarked'], 'numerical_feature': ['Age']}
```

Using automatic process for treating null values. This is done by `auto_process` method.

```python
>>> for _ in df_obj.auto_process():
         print(_)
The PassengerId column is skipped as it didn't have any kind of null values
The Survived column is skipped as it didn't have any kind of null values
The Pclass column is skipped as it didn't have any kind of null values
The Name column is skipped as it didn't have any kind of null values
The Sex column is skipped as it didn't have any kind of null values
The Age column is filled with the value of 24.0
The SibSp column is skipped as it didn't have any kind of null values
The Parch column is skipped as it didn't have any kind of null values
The Ticket column is skipped as it didn't have any kind of null values
The Fare column is skipped as it didn't have any kind of null values
The Embarked column is filled with the value of S

>>> df_obj.isnull_sum()
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
```

Treating null values manually by using `manual_process` method. Return type is `None`. it'll directly change into the main dataset.

```python
>>> df_obj.manual_process(column_name = 'Embarked', parameter = 'mode')      # Filling null values of 'Embarked' feature with its mode.
>>> df_obj.manual_process(column_name = 'Age', parameter = 'mean')           # Filling null values of 'Age' feature with its mean.
>>> df_obj.manual_process(column_name = 'Ticket', parameter = 'drop')        # Dropping Tickect column.
```

Removing the id column from the dataset by using `id_remover`method. Although it removes the id column from the dataframe but also it returns the id column that can be saved for later use.

```python
>>> df_obj.id_remover()
     PassengerId
0              1
1              2
2              3
3              4
4              5
..           ...
886          887
887          888
888          889
889          890
890          891

[891 rows x 1 columns]

>>> df_obj.head()
   Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Embarked
0         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500        S
1         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833        C
2         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250        S
3         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000        S
4         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500        S
```

Filling the null values for any particular column by the value passed as a parameter in `null_filler` method.

```python
>>> df_obj.null_filler(column_name = 'Embarked', value = 'S')
```

Getting the original dataset (preserved without any changes.) by using `get_data`.

```python
>>> df_obj.get_data
     PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0              1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1              2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2              3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3              4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4              5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
..           ...       ...     ...                                                ...     ...   ...    ...    ...               ...      ...   ...      ...
886          887         0       2                              Montvila, Rev. Juozas    male  27.0      0      0            211536  13.0000   NaN        S
887          888         1       1                       Graham, Miss. Margaret Edith  female  19.0      0      0            112053  30.0000   B42        S
888          889         0       3           Johnston, Miss. Catherine Helen "Carrie"  female   NaN      1      2        W./C. 6607  23.4500   NaN        S
889          890         1       1                              Behr, Mr. Karl Howell    male  26.0      0      0            111369  30.0000  C148        C
890          891         0       3                                Dooley, Mr. Patrick    male  32.0      0      0            370376   7.7500   NaN        Q

[891 rows x 12 columns]
```

> Apart from this there is also a methods like `concat`--> To concat dataframes/ series, `dataframe` --> To create a dataframe.
> Also if anyone add any kind of new method/class the test case for the same must be add in its respective test file.
> If anyone changes any of the method then one should run all test cases via `python -m unittest <test_file_name>` before pushing.
