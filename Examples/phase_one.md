# Data Preprocessing

> This file is only for example purpose where this shows the usage of different classes and methods avalaible in automation pakage. 1st phase only consists of data preprocessing phase.

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
