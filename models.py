from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor, multilayer_perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def linear_regression(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)      #X_test ---- remaining 
    print('R2 score--> ', regressor.score(X_test, y_test))          #y_test ---- remaining
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
    
    def model_num(self):
        if self.model_list == None:
            
            return "No model selected; model_list got None Attribute"
        elif type(self.model_list) != list:
            
            return f"Could not recognize {type(self.model_list)} object; List object must be passed"
        else:
            
            model_list_len = len(self.model_list)
            print("Working on it! please wait for a while")
            for model_list_index in range(model_list_len):
                
                if self.model_list[model_list_index] == 'LinearRegression':
                    
                    pred_output = linear_regression(self.X_train, self.y_train)
                    self.dict['LinearRegression'] = pred_output


print(Models("x", "y", ["LinearRegression"]).model_num())


# function for finding different scores ---- remaining
# other models function ---- remaining
# function for self.dict so that it should return y_pred according the the value passed by the user ---- remaining
