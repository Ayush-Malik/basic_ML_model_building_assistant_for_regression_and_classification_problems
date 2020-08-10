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
    def __init__(self, X_train, y_train, model_list = None):
        self.X_train = X_train
        self.y_train = y_train
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
