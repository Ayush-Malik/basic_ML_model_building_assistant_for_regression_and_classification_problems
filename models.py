from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor, multilayer_perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_squared_log_error
from streamlit import *


models_mapper = {
    # Regressors
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'SVR': SVR(),
    'MLPRegressor': MLPRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'XGBRegressor': XGBRegressor(),

    #Classifiers
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'SVC': SVC(),
    'MLPClassifier': MLPClassifier(),
    'DecisionTreeClassifier()': DecisionTreeClassifier(),
    'XGBClassifier': XGBClassifier(),
}


def Model_Trainer(Model, model_name, problem, X, y):
    X_train, X_test = X[0], X[1]
    y_train, y_test = y[0], y[1]
    
    Model.fit(X_train, y_train)
    y_pred = Model.predict(X_test)
    info(model_name, ':-')

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


    def model_call(self):
        '''
        model_call makes different function calls according to the Attributes received through model_list.
        '''
        if self.model_list == None:
            # To track for model_list is empty or not
            error("No model selected; model_list got None Attribute")
        
        elif type(self.model_list) != list:
            # To track type for model_list
            error(f"Could not recognize {type(self.model_list)} object; List object must be passed")
        
        else:
            # For calling respective functions according to the model_list
            text("")
            success("Working On It! Please Wait For a While")
            text("")
            text("")
            
            for model_name in self.model_list:
                Model = models_mapper[model_name]
                pred_output = Model_Trainer(
                    Model, model_name,
                    self.problem, self.X, self.y
                )
                self.dict[model_name] = pred_output


    def output(self, value):
        # Return the y_pred according to the value provided.
        return self.dict[value]


def acc_measure_cls(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    write("Accuracy Score:- ", acc)
    write("F1 Score:- ", f1)
    write("ROC AUC Score:- ", roc_auc)

def acc_measure_reg(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    write("R2 Score:- ", r2)
    write("Mean Squared Error:- ", mse)
    write("Mean Squared Log Error:- ", msle)


# print(Models("x", "y", ["LinearRegression"]).model_call())
