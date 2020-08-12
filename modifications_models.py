from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor, multilayer_perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_squared_log_error
from streamlit import *



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



models_mapper = {
    # Regressors
    'LinearRegression'         : LinearRegression() , 
    'RandomForestRegressor'    : RandomForestRegressor() ,
    'AdaBoostRegressor'        : AdaBoostRegressor() , 
    'SVR'                      : SVR() , 
    'MLPRegressor'             : MLPRegressor() , 
    'DecisionTreeRegressor'    : DecisionTreeRegressor() , 
    'XGBRegressor'             : XGBRegressor() , 

    #Classifiers
    'LogisticRegression'       : LogisticRegression() , 
    'RandomForestClassifier'   : RandomForestClassifier() , 
    'AdaBoostClassifier'       : AdaBoostClassifier() , 
    'SVC'                      : SVC() , 
    'MLPClassifier'            : MLPClassifier() , 
    'DecisionTreeClassifier()' : DecisionTreeClassifier() , 
    'XGBClassifier'            : XGBClassifier() , 
}

def mlp_classifier(X_train, X_test, y_train, y_test):
    regressor = MLPClassifier()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    subheader("MLP Classifier:-")
    acc_measure_cls(y_test, y_pred)
    return y_pred


def Model_Trainer(model_name , X_train , X_test , y_train , y_test):
    print('--> ' , model_name)
    Model.fit(X_train, y_train)
    y_pred =Model.predict(X_test)
    
    if 'Reg' in model_name and 'LogisticRegression' not in model_name:
        acc_measure = acc_measure_reg(y_test , y_pred)
    else:
        acc_measure = acc_measure_cls(y_test , y_pred)
    return y_pred


class Models:
    def __init__(self, X_list, y_list, model_list = None):
        '''
        Models is used to train different models on the given parameters.
        X_list:- list for X_train and X_test, order is important.
        y_list:- list for y_train and y_test, order is important.
        model_list:- takes in different model names, must pass in a list object.
        
        Examples
        ========
        >>> Models([X_train, X_test], [y_train, y_test], ["LinearRegression", "RandomForestRegressor"])
        >>> Models([X_train, X_test], [y_train, y_test], ["DecisionTreeClassifier", "RandomForestClassifier"])
        '''
        self.X_train = X_list[0]
        self.X_test = X_list[1]
        self.y_train = y_list[0]
        self.y_test = y_list[1]
        self.model_list = model_list
        self.dict = dict()


    def model_call(self):
        '''
        model_call makes different function calls according to the Attributes received through model_list.
        '''
        if self.model_list == None:
            # To track for model_list is empty or not
            return "No model selected; model_list got None Attribute"
        
        elif type(self.model_list) != list:
            # To track type for model_list
            return f"Could not recognize {type(self.model_list)} object; List object must be passed"
        
        else:
            # For calling respective functions according to the model_list
            model_list_len = len(self.model_list)
            text("")
            subheader("Working On It! Please Wait For a While")
            text("")
            text("")

            for model_name in self.model_list:
                Model = models_mapper[model_name]
                pred_output = Model_Trainer(
                    Model        , model_name , 
                    self.X_train , self.X_test,
                    self.y_train , self.y_test
                )
                self.dict['LinearRegression'] = pred_output
            
            
          

    def output(self, value):
        # Return the y_pred according to the value provided.
        return self.dict[value]




# print(Models("x", "y", ["LinearRegression"]).model_call())


# function for finding different scores ---- remaining
# function for self.dict so that it should return y_pred according the the value passed by the user ---- remaining
