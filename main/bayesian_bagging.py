import time
import pickle
import gc
from statistics import mean, stdev
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
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
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns


def introduce_jitter(data, feature, spread=0.25):
    """
    This function creates a pandas Series which introducesa jitter in a categorical
    feature of a pandas Dataframe to make scatter plots easier to visualize.

    Parameters
    ----------
    data: pandas DataFrame
        the Dataframe containing the column we want to transform
    feature: string
        the names of the column to transform

    spread: float
     This parameters controls the spread of the jitter

    Yields
    ------
    data[feature]: pandas Series
        Pandas series of floats rapresenting the categorical variable
    """
    temp = data[feature]*89
    for i, element in enumerate(temp.unique()):
        for j, value in enumerate(temp):
            if temp.iloc[j] == element:
                temp.iloc[j] = i+0.875 + np.random.random()*spread

    return (temp)


class MDIFeatureSelection(BaseEstimator, TransformerMixin):
    """
    Feature selection tecnique using random forest MDI
    """

    def __init__(self, n_features=None, max_features=int(1), n_estimators=200):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.n_features = n_features

    def fit(self, X, y):

        # fit a random forest classifier on training set
        clf = RandomForestClassifier(max_features=self.max_features,
                                     n_estimators=self.n_estimators,
                                     max_samples=0.10)
        # save results for later
        self.forest = clf.fit(X, y)
        return self

    def transform(self, X, y=None):

        # create array containing MDI feature importance
        importances = self.forest.feature_importances_

        # compute standard deviation of each feature
        std = np.std([tree.feature_importances_ for tree in self.forest.estimators_],
                     axis=0)
        imp_and_std = importances + std

        # set a treshold to evaluete if a feature is important or not
        treshold = 1/X.shape[1]
        X_ = X.copy()  # create a copy of original array

        # this list will contain indices of least important features
        sv = []

        # while calling the class if the parameter n_features was not set the least
        # important feature are considered those who have an MDI score less than the treshold
        if self.n_features is None:
            for i, value in enumerate(imp_and_std):
                if value < treshold:
                    sv.append(i)

            # sort list in encreasing order otherwise error can occour while deleting features
            sort_sv = sorted(sv, reverse=True)

            # remove least important features from data
            X_ = np.delete(X_, sort_sv, 1)

        # while calling the class if the parameter n_features was set
        # keep only a number of features equal to n_features
        # the ones to keep are those having the highes MDI importance score

        else:

            num_del_feat = X_.shape[1]-self.n_features

            # sort indexes of importance score in increasing oreder
            index = np.argsort(importances)

            # save indexes of least important features
            index_to_remove = index[0:num_del_feat]

            # sort indexes of least important features
            index_to_remove = np.sort(-index_to_remove)
            index_to_remove = -index_to_remove

            # remove least important features from data
            X_ = np.delete(X_, index_to_remove, 1)

        return X_

    def plot(self, X, y=None):

        importances = self.forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.forest.estimators_],
                     axis=0)
        column_names = X.columns

        # create a dataframe containig the feature names,value of importances and std dev
        dictionary = {"features": column_names, "importance": importances, "standard_dev": std}
        imp_data = pd.DataFrame(dictionary)
        imp_data = imp_data.sort_values(by=["importance"], ascending=False)  # sort array

        most_imp = imp_data.iloc[0:30, :]  # select 30 most important features
        least_imp = imp_data.iloc[:100:-1, :]  # select 30 most important features

        # plot barplot of most important features
        most_imp.plot(
            kind="bar",
            x="features",
            y="importance",
            yerr="standard_dev"
        )
        plt.title("Most important features")

        tresh = 1 / imp_data.shape[0]  # compute treshold
        # plot an horizontal line that rapresents the treshold
        plt.axhline(y=tresh, linestyle="--")
        plt.show()

        least_imp.plot(
            kind="bar",
            x="features",
            y="importance",
            yerr="standard_dev"
        )
        plt.title("least important features")

        tresh = 1 / imp_data.shape[0]  # compute treshold
        # plot an horizontal line that rapresents the treshold
        plt.axhline(y=tresh, linestyle="--")
        plt.show()


if __name__ == "__main__":

    NAIVE = False
    BAGGING_25 = False
    BAGGING_50 = False
    PLOT = True

    if NAIVE is True:

        #import dataset
        data = import_training_set()

        # create a dataframe withouth the class label
        X = data.drop(["action"], axis=1)
        # create an array the rapresent the class label
        y = np.ravel(data.loc[:, ["action"]])
        y = np.array(y)

        # create a pipeline containing all ML steps
        pipe = Pipeline([('miss_value', SimpleImputer()),
                         ('scaler', StandardScaler()),
                         ('reduce_dim', 'passthrough'),
                         ('clf', GaussianNB())])

        # create a param grid
        param_grid = [
            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [PCA()],
             'reduce_dim__n_components': [0.91, 0.93, 0.95, 0.97]},

            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [SelectKBest()],
             'reduce_dim__k': [20, 30, 40, 50]},


            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': ['passthrough']},

            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [MDIFeatureSelection()]},

            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [MDIFeatureSelection()],
             'reduce_dim__n_features': [30, 40, 50]}]

        grid = GridSearchCV(pipe,
                            cv=cv,
                            n_jobs=1,
                            param_grid=param_grid,
                            scoring='f1',
                            verbose=2)
        grid.fit(X, y)
        print(grid)

        # save grid search results
        pd = pd.DataFrame(grid.cv_results_)
        pd.to_csv("Results/naive_bayes.csv")

        # print the best parameters
        print(grid.best_params_)

    if BAGGING_25 is True:

        data = import_training_set()
        X = data.drop(["action"], axis=1)
        # create an array the rapresent the class label
        y = np.ravel(data.loc[:, ["action"]])
        y = np.array(y)

        bagging = BaggingClassifier(GaussianNB(),
                                    bootstrap=True,
                                    max_samples=0.10,
                                    n_estimators=25)

        pipe = Pipeline([('miss_value', SimpleImputer()),
                         ('scaler', StandardScaler()),
                         ('reduce_dim', 'passthrough'),
                         ('clf', bagging)])

        param_grid = [
            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [PCA()],
             'reduce_dim__n_components': [0.91, 0.93, 0.95, 0.97]},

            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [SelectKBest()],
             'reduce_dim__k': [20, 30, 40, 50]},


            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': ['passthrough']}]

        grid = GridSearchCV(pipe,
                            n_jobs=1,
                            param_grid=param_grid,
                            scoring='f1',
                            verbose=2)
        grid.fit(X, y)
        print(grid)

        pd = pd.DataFrame(grid.cv_results_)
        pd.to_csv("Results/bagging_25_outliers.csv")

        print(grid.best_params_)

    if BAGGING_50 is True:

        data = import_training_set()
        X = data.drop(["action"], axis=1)
        # create an array the rapresent the class label
        y = np.ravel(data.loc[:, ["action"]])
        y = np.array(y)

        bagging = BaggingClassifier(GaussianNB(),
                                    bootstrap=True,
                                    max_samples=0.10,
                                    n_estimators=50)

        pipe = Pipeline([('miss_value', SimpleImputer()),
                         ('scaler', StandardScaler()),
                         ('reduce_dim', 'passthrough'),
                         ('clf', bagging)])

        param_grid = [
            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [PCA()],
             'reduce_dim__n_components': [0.91, 0.93, 0.95, 0.97]},

            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': [SelectKBest()],
             'reduce_dim__k': [20, 30, 40, 50]},


            {'miss_value__strategy': ['constant', 'mean'],
             'reduce_dim': ['passthrough']}]

        grid = GridSearchCV(pipe,
                            n_jobs=1,
                            param_grid=param_grid,
                            scoring='f1',
                            verbose=2)
        grid.fit(X, y)
        print(grid)

        pd = pd.DataFrame(grid.cv_results_)
        pd.to_csv("Results/bagging_50_outliers.csv")

        print(grid.best_params_)

    if PLOT is True:

        # load first grid search results
        data = pd.read_csv('Results/naive_outliers.csv')
        # delete some duplicate rows from dataframe
        a = [x for x in range(16, 24)]
        data = data.drop(a, axis=0)
        data = data.drop([27, 26], axis=0)

        # load second grid search result
        data1 = pd.read_csv('Results/naive_nice_removed.csv')
        data1["param_miss_value__strategy"] = 'removed'

        # merge the two dataframe
        df = pd.concat([data, data1])
        print(df.param_reduce_dim)

        # introduce jitter
        df["param_reduce_dim_jitter"] = introduce_jitter(df, "param_reduce_dim")

        g = sns.scatterplot(y="mean_test_score",
                            x="mean_fit_time",
                            style="param_miss_value__strategy",
                            hue="param_reduce_dim",
                            palette="gnuplot",
                            legend="brief",
                            data=df)

        g.set_title("Naive Bayes", fontsize=20)
        g.set_xlabel("Computational time [sec]")
        g.set_ylabel("Mean f1 score on test set")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

        # load first grid search results
        data_25 = pd.read_csv('Results/bagging_naive_25_1.csv')

        # load second grid search result
        data_25_1 = pd.read_csv('Results/bagging_naive_25_2.csv')

        # merge the two dataframe
        df_25 = pd.concat([data_25, data_25_1])
        print(df_25.columns)

        # introduce jitter
        df_25["param_reduce_dim_jitter"] = introduce_jitter(df_25, "param_reduce_dim")

        g_25 = sns.scatterplot(x="mean_fit_time",
                               y="mean_test_score",
                               hue="param_reduce_dim",
                               size="param_miss_value__strategy",
                               palette="gnuplot",
                               style="param_clf__max_features",
                               legend="brief",
                               data=df_25)

        g_25.set_title("Naive bagging 25", fontsize=20)
        g_25.set_xlabel("Computational time [sec]")
        g_25.set_ylabel("Mean f1 score on test set")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()
