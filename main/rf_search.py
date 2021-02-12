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
import matplotlib.pyplot as plt
import seaborn as sns
from bayesian_bagging import introduce_jitter


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
        for j, element in enumerate(outliers):
            if element == -1:
                np.delete(X_, j, 0)
                i = i+1
        print('total number of outliers is: {}'.format(i))
        return X_


if __name__ == '__main__':
    SEARCH = False
    PLOT = False
    NO_OUTLIERS = False
    RESULTS = False
    END = True

    if SEARCH is True:
        data = import_training_set()

        forest = RandomForestClassifier(n_estimators=500,
                                        min_weight_fraction_leaf=0.05,
                                        class_weight='balanced_subsample',
                                        max_samples=0.10,
                                        max_features=3,
                                        bootstrap=True)

        pipe = Pipeline([('miss_values', SimpleImputer()),
                         ('scaler', 'passthrough'),
                         ('reduce_dim', 'passthrough'),
                         ('clf', forest)])

        variance = [x/100 for x in range(85, 101, 2)]
        print(variance)
        param_grid = [
            {'miss_values__strategy': ["constant"],
             'miss_values__fill_value': [-999],
             'scaler': [StandardScaler()],
             'reduce_dim': [PCA()],
             'reduce_dim__n_components': variance}]

        cv = TimeSeriesSplit(
            n_splits=3,
            test_size=400000,
            gap=200000)

        X = data.drop(["action"], axis=1)
        # create an array the rapresent the class label
        y = np.ravel(data.loc[:, ["action"]])
        y = np.array(y)

        grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv=cv,
                            scoring='f1', verbose=2, return_train_score=True)
        grid.fit(X, y)

        pd = pd.DataFrame(grid.cv_results_)
        pd.to_csv("Results/rf_PCA1.csv")

        print(grid.best_params_)

    if PLOT is True:
        results = pd.read_csv("Results/rf_PCA.csv")
        print(results[["mean_test_score", "mean_train_score"]])

        results1 = pd.read_csv("Results/rf_PCA_nice.csv")
        print(results1[["mean_test_score", "mean_train_score"]])

        xx = results['param_reduce_dim__n_components']
        y1 = results["mean_test_score"]
        dy1 = results["std_test_score"]
        y2 = results["mean_train_score"]
        dy2 = results["std_train_score"]
        y3 = results1["mean_test_score"]
        dy3 = results1["std_test_score"]
        y4 = results1["mean_train_score"]
        dy4 = results["std_train_score"]

        # plt.errorbar(xx, y1, yerr=dy1, label='-999')
        # plt.errorbar(xx, y2, yerr=dy2, label='-999')
        ax = results1.plot(kind='bar',
                           x='param_reduce_dim__n_components',
                           y="mean_test_score",
                           yerr="std_test_score",
                           position=0,
                           width=0.3,
                           color='black')

        results1.plot(kind='bar',
                           x='param_reduce_dim__n_components',
                           y="mean_train_score",
                           yerr="std_train_score",
                           ax=ax,
                           position=1.2,
                           width=0.3,
                           color='blue')

        results1.plot(kind='bar',
                      x='param_reduce_dim__n_components',
                      y="mean_fit_time",
                      yerr="std_fit_time")
        print(results1.columns)
        # plt.errorbar(xx, y4, yerr=dy4, label='train')
        plt.legend()
        plt.show()

    if NO_OUTLIERS is True:
        data = import_training_set()

        X = data.drop(["action"], axis=1)
        # create an array the rapresent the class label
        y = np.ravel(data.loc[:, ["action"]])
        y = np.array(y)

        splits = folds.split(X, y)

        folds = TimeSeriesSplit(n_splits=3,
                                test_size=400000,
                                gap=200000)

        score_cv = []  # empty list will contain accuracy score of each split

        forest = RandomForestClassifier(n_estimators=500,
                                        min_weight_fraction_leaf=0.05,
                                        class_weight='balanced_subsample',
                                        max_samples=0.10,
                                        max_features=3,
                                        bootstrap=True)

        # start cross validation
        for fold_n, (train_index, val_index) in enumerate(splits):
            print('Fold: {}'.format(fold_n+1))
            # for each iteration define the boundaries of training and validation set
            X_train = X.iloc[train_index[0]:train_index[-1], :]
            X_val = X.iloc[val_index[0]:val_index[-1], :]
            y_train = y[train_index[0]:train_index[-1]]
            y_val = y[val_index[0]:val_index[-1]]

            # fit model and compute score

            imp = SimpleImputer(strategy='constant', fill_value=0)
            imp.fit(X_train, y_train)
            imp.transform(X_train, y_train)

            scaler = StandardScaler()
            scaler.fit(X_train, y_train)
            scaler.transform(X_train, y_train)

            pca = PCA(n_components=0.90)
            pca.fit(X_train, y_train)
            pca.transform(X_train, y_train)

            del_outliers = RemoveOutliers()
            del_outliers.transform(X_train, y_train)

            forest.fit(X_train, y_train)

            imp.transform(X_val, y_val)
            scaler.transform(X_val, y_val)
            pca.transform(X_val, y_val)
            del_outliers.transform(X_val, y_val)

            predictions = forest.predict(X_val)
            accuracy = roc_auc_score(y_val, predictions)
            score_cv.append(accuracy)

            print("AUC: {}".format(accuracy))

            # delete train and validation set to save memory
            del X_train, X_val, y_train, y_val
            gc.collect()

    if RESULTS is True:
        forest1 = pd.read_csv("Results/rf_PCA1.csv")

        forest2 = pd.read_csv("rf_PCA_kaggle.csv")
        naive1 = pd.read_csv("bagging_naive_25.csv")
        naive2 = pd.read_csv("bagging_naive_25_2.csv")
        naive3 = pd.read_csv("naive (1).csv")

        print(naive1[["mean_test_score", "mean_train_score"]])
        print(naive2[["mean_test_score", "mean_train_score"]])
        print(naive3[["mean_test_score", "mean_train_score"]])

        print(forest1[["mean_test_score", "mean_train_score"]])

        print(forest2[["mean_test_score", "mean_train_score", "param_reduce_dim__n_components"]])

        data = import_training_set(rows=20000)

        forest = RandomForestClassifier(class_weight='balanced_subsample',
                                        max_samples=0.10,
                                        max_features='auto',
                                        bootstrap=True)

        pipe = Pipeline([('miss_values', SimpleImputer()),
                         ('classifier', forest)])

        estimators = range(100, 800, 50)
        depth = range(10, 80, 5)
        param_grid = [
            {'classifier__n_estimators': estimators,
             'classifier__max_depth': depth}]

        cv = TimeSeriesSplit(
            n_splits=3,
            test_size=1500,
            gap=1500)

        X = data.drop(["action"], axis=1)
        # create an array the rapresent the class label
        y = np.ravel(data.loc[:, ["action"]])
        y = np.array(y)

        grid = GridSearchCV(pipe, n_jobs=-1, param_grid=param_grid, cv=cv,
                            scoring='f1', verbose=2, return_train_score=True)
        grid.fit(X, y)

        pd = pd.DataFrame(grid.cv_results_)
        pd.to_csv("Results/rf_PCA1.csv")

        print(grid.best_params_)

    if END is True:

        data = pd.read_csv("Results/rf_PCA1.csv")

        print(data.columns)
        data["max depth"] = introduce_jitter(data, "param_classifier__max_depth", spread=0.33)

        g = sns.scatterplot(x="param_classifier__max_depth",
                            y="mean_test_score",
                            hue="param_classifier__n_estimators",
                            palette="bone",
                            size=20,
                            data=data)
        plt.show()

        g1 = sns.scatterplot(x="param_classifier__max_depth",
                             y="mean_train_score",
                             hue="param_classifier__n_estimators",
                             palette="bone",
                             size=20,
                             data=data)

        plt.show()
