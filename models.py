from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor, multilayer_perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def linear_regression(X_train, X_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
    return y_pred


def random_forest_regressor(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
    return y_pred


def ada_boost_regressor(X_train, X_test, y_train, y_test):
    regressor = AdaBoostRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
    return y_pred


def svr(X_train, X_test, y_train, y_test):
    regressor = SVR()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
    return y_pred


def mlp_regressor(X_train, X_test, y_train, y_test):
    regressor = MLPRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
    return y_pred


def decision_tree_regressor(X_train, X_test, y_train, y_test):
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
    return y_pred


def xgb_regressor(X_train, X_test, y_train, y_test):
    regressor = XGBRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('R2 score--> ', regressor.score(X_test, y_test))
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
            print("Working on it! please wait for a while")
            
            for model_list_index in range(model_list_len):
                # For Regression problems
                if self.model_list[model_list_index] == 'LinearRegression':
                    pred_output = linear_regression(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                        )
                    self.dict['LinearRegression'] = pred_output
                
                if self.model_list[model_list_index] == 'RandomForestRegressor':
                    pred_output = random_forest_regressor(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                    )
                    self.dict['RandomForestRegressor'] = pred_output
                
                if self.model_list[model_list_index] == 'AdaBoostRegressor':
                    pred_output = ada_boost_regressor(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                    )
                    self.dict['AdaBoostRegressor'] = pred_output
                
                if self.model_list[model_list_index] == 'SVR':
                    pred_output = svr(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                    )
                    self.dict['SVR'] = pred_output
                
                if self.model_list[model_list_index] == 'MLPRegressor':
                    pred_output = mlp_regressor(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                    )
                    self.dict['MLPRegressor'] = pred_output
                
                if self.model_list[model_list_index] == 'DecisionTreeRegressor':
                    pred_output = decision_tree_regressor(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                    )
                    self.dict['DecisionTreeRegressor'] = pred_output
                
                if self.model_list[model_list_index] == 'XGBRegressor':
                    pred_output = xgb_regressor(
                        self.X_train, self.X_test,
                        self.y_train, self.y_test
                    )
                    self.dict['XGBRegressor'] = pred_output


print(Models("x", "y", ["LinearRegression"]).model_call())


# function for finding different scores ---- remaining
# classification models function ---- remaining
# function for self.dict so that it should return y_pred according the the value passed by the user ---- remaining
