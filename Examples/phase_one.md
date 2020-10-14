# Data Preprocessing

> This file is only for example purpose where this shows the usage of different classes and methods avalaible in automation package. 1st phase only consists of data preprocessing phase.

Importing libraries.

```python
>>> from automation import DataCleaner, FeatTransform, Outliers
>>> import pandas as pd
```

Loading the dataset and creating the object of `DataCleaner`.

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

Getting the unique values present in the dataset by using `unique` method.

```python
>>> df_obj.unique('Sex')
... ['male' 'female']
```

Getting the percentage of unique values count in the dataset by using `unique_prcntg` method.

```python
>>> df_obj.unique_prcntg('Sex')
... 0.22
```

Getting value counts for a particular column of the dataset by using `value_counts` method.

```python
>>> df_obj.value_counts('Sex')
male      577
female    314
Name: Sex, dtype: int64
```

Mapping the values of any particular column according to the dict object is passed and all of this done by `map` method.
**NOTE:- It not map the values in actual dataframe instead it only returns the column with mapped values.**

```python
>>> new_col = df_obj.map('Sex', {'male': 1, 'female': 0})
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
>>> df_obj.get_dummies('Ticket')
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

Getting the sum of isnull by using `isnull_sum` method.

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
```

Getting the total percentage of null values for a particular column by using `isnull_sum_prcntg` method.

```python
>>> df_obj.isnull_sum_prcntg('Cabin')
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
>>> df_obj.central_tendency_finder('Age', 'mean')
... 29.69911764705882

>>> df_obj.central_tendency_finder('Age', 'mode')
... 24.0

>>> df_obj.central_tendency_finder('Age', 'median')
... 28.0
```

Getting the inter quartile range for any particular feature by using `inter_quartile_range` method.

```python
>>> df_obj.inter_quartile_range('Age')
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
