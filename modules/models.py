from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor, multilayer_perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_squared_log_error
from streamlit import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


models_mapper = {
    # Regressors
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'SVR': SVR(),
    'MLPRegressor': MLPRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'XGBRegressor': XGBRegressor(),

    #Classifiers
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(),
    'MLPClassifier': MLPClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'XGBClassifier': XGBClassifier(),
}


param_test_dict = {
    "LogisticRegression()": {"penalty": ['l1', 'l2'],
                            "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            "solver": ['newton-cg', 'liblinear', 'saga'],
                            "max_iter": [100, 200, 300, 400]
                        },

    "XGBClassifier()": {"gamma": [i/10.0 for i in range(0, 5)],
                        "learning_rate": [0.05, 0.1, 0.2, 0.3],
                        "max_depth": [5, 6, 7, 9],
                        "min_child_weight": [3, 5, 6],
                        "n_estimators": [num for num in range(0, 1000, 100)],
                        },
}


def Model_Trainer(Model, model_name, problem, X, y, hypertunnig):
    X_train, X_test = X[0], X[1]
    y_train, y_test = y[0], y[1]

    Model.fit(X_train, y_train)
    y_pred = Model.predict(X_test)
    print(model_name)

    if hypertunnig == True:
        y_pred = ht_model_runner(X, y, problem, Model)
        # y_pred = HO_object.output()
        return y_pred
    elif hypertunnig == False:
        if problem.lower() == 'regression':
            acc_measure = acc_measure_reg(y_test, y_pred)
        else:
            acc_measure = acc_measure_cls(y_test, y_pred)
        return y_pred


class Models:
    def __init__(self, X_list, y_list, problem, model_list = None):
        '''
        Models is used to train different models on the given parameters.
        X_list:- list for X_train and X_test, order is important.
        y_list:- list for y_train and y_test, order is important.
        problem:- str object used to describe the problem statement.
        model_list:- takes in different model names, must pass in a list object.
        
        Examples
        ========
        >>> Models([X_train, X_test], [y_train, y_test], "Regression", ["LinearRegression", "RandomForestRegressor"])
        >>> Models([X_train, X_test], [y_train, y_test], "Classification", ["DecisionTreeClassifier", "RandomForestClassifier"])
        '''
        self.X = X_list
        self.y = y_list
        self.problem = problem
        self.model_list = model_list
        self.dict = dict()


    def model_call(self, hypertunnig=False):
        '''
        model_call makes different function calls according to the Attributes received through model_list.
        '''
        if self.model_list == None:
            # To track for model_list is empty or not
            print("No model selected; model_list got None Attribute")
        
        elif type(self.model_list) != list:
            # To track type for model_list
            print(f"Could not recognize {type(self.model_list)} object; List object must be passed")
        
        else:
            # For calling respective functions according to the model_list
            #success("Working On It! Please Wait For a While")
            print("")
            
            for model_name in self.model_list:
                Model = models_mapper[model_name]
                pred_output = Model_Trainer(
                    Model, model_name,
                    self.problem, self.X, self.y,
                    hypertunnig
                )
                self.dict[model_name] = pred_output


    def output(self, value):
        # Return the y_pred according to the value provided.
        return self.dict[value]


def acc_measure_cls(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred , average = 'micro')
    try:
        roc_auc = roc_auc_score(y_test, y_pred)
        print("Accuracy Score:- ", acc)
        print("F1 Score:- ", f1)
        print("ROC AUC Score:- ", roc_auc)
    except:
        print("Accuracy Score:- ", acc)
        print("F1 Score:- ", f1)
        print("ROC AUC Score can't be shown because target feature is of multiclass")


def acc_measure_reg(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    try:
        msle = mean_squared_log_error(y_test, y_pred)
        print("R2 Score:- ", r2)
        print("Mean Squared Error:- ", mse)
        print("Mean Squared Log Error:- ", msle)
    except:
        print("R2 Score:- ", r2)
        print("Mean Squared Error:- ", mse)
        print("MSLE can't be shown as there might be some negative values present in prediction dataset.")


# print(Models("x", "y", ["LinearRegression"]).model_call())
def feature_list(df):
    fe_list = []
    for col in df.columns:
        fe_list.append(col)
    return fe_list

def set_target(df, target_feature):
    # Suggestion from us
    if len(df[target_feature].unique()) < 10 :
        return("Classification", "Acc to us this is a Classification Problem 😁😁 \n .Rest is Your Choice. \n Ignore At your Own Risk 🤣🤣🤣")
    else:
        return("Regression", "Acc to us this is a Regression Problem 😁😁 \n .Rest is Your Choice. \n Ignore At your Own Risk 🤣🤣🤣")

def train_test_splitter(df, prcntage):

    train_rows = int( len(df) * prcntage )
    train = df.iloc[ : train_rows  : , : ]
    test  = df.iloc[ train_rows  : : , : ]

    return(train, test)

def x_y_maker(target_feature, train, test):
    y_train = train[target_feature].values
    x_train = train.drop(target_feature , axis = 1)
    x_train = x_train.values

    y_test = test[target_feature].values
    x_test = test.drop(target_feature , axis = 1)
    x_test = x_test.values
    return(x_train, x_test, y_train, y_test)


class HyperparameterOptimization:
    def __init__(self, X_list, y_list, problem, Model):
        self.X_train, self.X_test = X_list[0], X_list[1]
        self.y_train, self.y_test = y_list[0], y_list[1]
        self.problem = problem
        self.Model = Model
        self.param_test = param_test_dict[str(Model)]

    def scoring(self):
        if self.problem.lower() == 'regression':
            self.score = "r2"
        else:
            self.score = "accuracy"

    def param(self):
        pass

    def gridsearchcv(self):
        gsearch = GridSearchCV(estimator = self.Model,
                                param_grid = self.param_test,
                                scoring = self.score,
                                n_jobs = -1,
                                # iid = False,
                                verbose = 1,
                                # cv = 5
                            )
        gsearch.fit(self.X_train, self.y_train)
        self.best_estimator = gsearch.best_estimator_
        print(self.best_estimator)

    def ht_model(self):
        classifier = self.best_estimator
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)

    def acc_check(self):
        if self.problem.lower() == "regression":
            acc_measure = acc_measure_reg(self.y_test, self.y_pred)
        else:
            acc_measure = acc_measure_cls(self.y_test, self.y_pred)

    def output(self):
        return self.y_pred


def ht_model_runner(X, y, problem, Model):
    HO_object = HyperparameterOptimization(X, y, problem, Model)
    HO_object.scoring()
    HO_object.gridsearchcv()
    HO_object.ht_model()
    HO_object.acc_check()
    y_pred = HO_object.output()
    return y_pred
