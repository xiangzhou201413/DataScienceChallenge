#(1) Data preprocessing with _processing_data(). (2)Balance data with _balance_data() 
#(3)Transform data for machine learning with _transform_data(). (4)Screening for a classifier with clfs_search().
#(5)Tuning the selected classifier (XGb) with parameter_tuning() and generate scores.

from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import operator,sys
import random

class DefaultPredictions(object):
    def __init__(self):
        self.input_file = "default_of_credit_card_clients.xls"
        self.if_balance_data = True
        self._processing_data()
        self.X_train = []
        self.y_train = []
        self.js_label_mapper = {}
        self._transform_data()
        self.selected_clf = XGBClassifier(n_estimators = 120, max_depth = 5)

    def _processing_data(self):
        self.all_data = pd.read_excel(self.input_file, skiprows = [1])
        all_col_names = [col for col in self.all_data]
        all_variables = all_col_names[:-1]
        self.label_symbol = all_col_names[-1]
        print(all_variables, self.label_symbol)
        label_list = list(self.all_data[self.label_symbol])
        self.default_ct = label_list.count(1)
        print(self.default_ct)
        if self.if_balance_data:
            self._balance_data()
        else:
            self.selected_data = self.all_data
        self.train_data = self.selected_data
        self.category_names = ['X2','X3','X4']
        self.numerical_names = [col for col in self.train_data if col != self.label_symbol and col not in self.category_names]
        print("Finished processing data.")
        
    def _balance_data(self):
        not_default_index = list(self.all_data.loc[self.all_data[self.label_symbol] == 0].index)
        not_default_select = random.sample(not_default_index, self.default_ct)
        self.selected_data = self.all_data.drop(self.all_data.index[not_default_select])
        print("Some of balanced data:")
        print(self.selected_data[:30])

    def _encoding_category_data(self,data):
        print("Transforming category data ...")
        encoded_categories_data = {}
        for cat in self.category_names:
            cat_data = data[[cat]].copy()
            ohe = OneHotEncoder(sparse=False, categories='auto')
            data_transformed = list(ohe.fit_transform(cat_data))
            encoded_categories_data.update({cat:data_transformed})
        return encoded_categories_data

    def clfs_search(self):
        test_size = 2000
        X_train = self.X_train[:-test_size]
        y_train = self.y_train[:-test_size]
        X_test = self.X_train[-test_size:]
        y_test = self.y_train[-test_size:]

        classifiers = {'Xgboost':  XGBClassifier(n_estimators = 200),
                       'Ridge':    RidgeClassifier(),
                       'AdaBoost': AdaBoostClassifier(n_estimators = 100),
                       'Neural network': MLPClassifier(hidden_layer_sizes=(300,)),
                       'Random F': ExtraTreesClassifier(n_estimators = 200)}
        for index, (name, classifier) in enumerate(classifiers.items()):
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy for %s: %0.1f%% " % (name, accuracy * 100))

    def parameter_tuning(self):
        clf = XGBClassifier()
        params={'max_depth': [5,7,9,11],'n_estimators': [120,220,320]}
        rs = GridSearchCV(clf, params,cv=5)
        rs.fit(self.X_train, self.y_train)
        best_est = rs.best_estimator_
        print("Best Parameters:")
        print(best_est.max_depth,best_est.n_estimators)

        k_fold = KFold(n_splits=5)
        cv_score = cross_val_score(best_est, self.X_train, self.y_train, cv=k_fold, n_jobs=-1)
        print("cross validation scores:")
        print(cv_score)

    def _transform_data(self):
        encoded_categories_train = self._encoding_category_data(self.train_data)

        labels = list(self.train_data[self.label_symbol])
        unique_lbs = set(labels)
        for i, u in enumerate(unique_lbs):
            self.js_label_mapper.update({u:i})
        for i, js in enumerate(labels):
            one_vec = []
            lb = self.js_label_mapper[js]
            self.y_train.append(lb)
            for name in self.numerical_names:
                num_data = list(self.train_data[name])[i]
                one_vec.append(num_data)
            for name in self.category_names:
                cat_data = encoded_categories_train[name][i]
                one_vec += list(cat_data)
            self.X_train.append(one_vec)

        print(self.X_train[0],self.y_train[0])            

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

if __name__ == '__main__':
    predict = DefaultPredictions()
#   predict.clfs_search()
    predict.parameter_tuning()
