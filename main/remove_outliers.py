from sklearn.base import BaseEstimator, TransformerMixin
import time
import pickle
import gc
from statistics import mean, stdev
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from initial_import import import_training_set
from splitting import split_data
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor


data = import_training_set(rows=100000)
'''
forest = RandomForestClassifier(n_estimators=500,
                                min,
                                class_weight='balanced_subsample',
                                max_samples=0.10,
                                max_features=3,
                                bootstrap=True)

pipe = Pipeline([('miss_values', SimpleImputer()),
                 ('scaler', 'passthrough'),
                 ('reduce_dim', 'passthrough'),
                 ('clf', forest)])


param_grid = [
    {'miss_values__strategy': ["mean", "constant"],
     'scaler': [StandardScaler(), RobustScaler()],
     'reduce_dim': [PCA()],
     'reduce_dim__n_components': [0.85, 0.90, 0.95]}]

cv = TimeSeriesSplit()

X = data.drop(["action"], axis=1)
# create an array the rapresent the class label
y = np.ravel(data.loc[:, ["action"]])
y = np.array(y)


grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv=cv, scoring='f1', verbose=2)
grid.fit(X, y)

pd = pd.DataFrame(grid.cv_results_)
pd = pd.sort_values(by="mean_test_score", ascending=False)
pd.to_csv("Results/rf_1.csv")

print(grid.best_params_)
'''


class RemoveOutliers(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        print('hello')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        lof = LocalOutlierFactor()
        X_ = X.copy()
        outliers = lof.fit_predict(X_)
        i = 0
        for element in outliers:
            if element == -1:
                for j in range(X_.shape[1]):
                    X_[i, j] = -999
                i = i+1
        print('total number of outliers is: {}'.format(i))
        return X_


X, y, X_test, y_test = split_data(data, no_fill=True)

rem = RemoveOutliers()
forest = RandomForestClassifier(n_estimators=10)

pipe = Pipeline([('miss_values', SimpleImputer()),
                 ('outliers', rem),
                 ('clf', forest)])

pipe1 = Pipeline([('miss_values', SimpleImputer()),

                  ('clf', forest)])
pipe.fit(X, y)
pipe1.fit(X, y)
print(pipe.score(X_test, y_test))
print(pipe1.score(X_test, y_test))
