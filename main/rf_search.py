<<<<<<< HEAD
"""
This is the main module we used to fit the models. After defining a pipeline and a
param grid, a grid search with CV is initialized and all the important informations
about the search are saved in a location choosen by the user.
"""
import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from initial_import import import_training_set


def search(data, pipe, param_grid,filepath=None,cv=None):
    """
    This is the main function we used to do the hyperparameter searches. We set up
    grid search with a TimeSeries CV and then we save all the useful informations about
    the search, such as the f1 score on the test and training set, the computational time
    etc, as a csv file.

    Parameters
    ----------
    data: DataFrame
        The dataset used to fit the model
    pipe: Pipeline
        The pipeline
    param_grid: dict
        The param grid with the hyperparameters we want to test
    filepath: string (default=None)
        The filepath in which the dataframe containig all the informations about the
        search is saved
    cv: CV
        The cross validation parameters

    Yields
    ------
    df: pd.DataFrame
        A DataFrame including all the information about the GridSearch

    """

    # separate class label from the rest of the dataset
    X = data.drop(["action"], axis=1)
    y = np.ravel(data.loc[:, ["action"]])
    y = np.array(y)
    # set cross validation
    if cv is None:
        cv = TimeSeriesSplit(n_splits=10, test_size=100000, gap=100000)
    # set hyperparameter search
    grid = GridSearchCV(pipe,param_grid=param_grid,cv=cv,n_jobs=1,scoring='f1',
                        verbose=2,return_train_score=True)
    # fit the search
    grid.fit(X, y)
    # save search results as pandas DataFrame
    df = pd.DataFrame(grid.cv_results_)
    if filepath is not None:
        df.to_csv(filepath)
    # print best performing model
    print("the best hyperparameters are:")
    print(grid.best_params_)
    # delete X and y
    del X, y
    gc.collect()

    return df

if __name__ == '__main__':
    # load dataset
    data = import_training_set(fast_pc = True)
    data.dropna(inplace=True)
    # set up classifier and pipeline
    bagging = BaggingClassifier(base_estimator=GaussianNB(),
                                n_estimators=25,
                                bootstrap=True,
                                max_samples=0.25,
                                n_jobs=1)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('reduce_dim', 'passthrough'),
                     ('clf', bagging)])

    # set up param grid
    param_grid = [
            {'reduce_dim': ['passthrough']},

            {'reduce_dim': [PCA()],
             'reduce_dim__n_components': [0.91, 0.93, 0.95, 0.97],
             'clf__max_features' : [0.33,0.66,1.0]},

            {'reduce_dim': [SelectKBest()],
             'reduce_dim__k': [20, 30, 40, 50],
             'clf__max_features' : [0.33,0.66,1.0]}]

    cv = TimeSeriesSplit(n_splits=10, test_size=100000, gap=100000)
    # initiate hyperparameter search
    results = search(data,pipe,param_grid,filepath='Results/bagging_naive_25_4.csv',cv=cv)
=======
"""
This is the main module we used to fit the models
At the beginning of the main 3 boolean variables can be set

Search:
    The most important part of this module. After defining a pipeline and a
    param grid, a grid search with CV is initialized and all the important informations
    about the search are saved in a location choosen by the user.

Forest:
    This module is used to visualize the results of the grid searches for the random
    forest algorithm.

Naive:
    This module is used to visualize the results of the grid searches for the naive
    bayes classifier.
"""


import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from initial_import import import_training_set


def search(data, pipe, param_grid, filepath, splits=10, size=100000, gap=100000):
    """
    This is the main function we used to do the hyperparameter searches. We set up
    grid search with a TimeSeries CV and then we save all the useful informations about
    the search, such as the f1 score on the test and training set, the computational time
    etc, as a csv file.

    Parameters
    ----------
    data: DataFrame
        The dataset used to fit the model
    pipe: Pipeline
        The pipeline
    param_grid: dict
        The param grid with the hyperparameters we want to test

    filepath: string
        the name used to save the dataframe containig all the informations about the
        search
    splits: int (default 10)
        The number of splits used for the CV
    size: int (default 100000)
        The maximum test size for each CV split
    gap: int (default 100000)
        Gap between train and test set

    Yields
    ------
    A csv file, saved in the location indicated by the filepath

    """

    # separate class label from the rest of the dataset
    X = data.drop(["action"], axis=1)
    y = np.ravel(data.loc[:, ["action"]])
    y = np.array(y)

    # set cross validation
    cv = TimeSeriesSplit(n_splits=splits, test_size=size, gap=gap)

    # set hyperparameter search
    grid = GridSearchCV(pipe,
                        param_grid=param_grid,
                        cv=cv,
                        n_jobs=1,
                        scoring='f1',
                        verbose=1,
                        return_train_score=True)
    # fit the search
    grid.fit(X, y)

    # save search results as csv
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(filepath)

    # print best performing model
    print("the best hyperparameters are:")
    print(grid.best_params_)

    # delete X and y
    del X, y
    gc.collect()


if __name__ == '__main__':
    SEARCH = True
    FOREST = False
    NAIVE = False

    if SEARCH is True:

        # load dataset
        data = import_training_set(fast_pc = True)

        # set up classifier and pipeline
        forest = RandomForestClassifier(class_weight='balanced_subsample',
                                        max_samples=0.10,
                                        max_features='auto',
                                        bootstrap=True,
                                        n_jobs=-1)

        pipe = Pipeline([('miss_values', SimpleImputer()),
                         ('clf', forest)])

        # set up param grid
        param_grid = [{'clf__n_estimators': range(100, 700, 200),
                       'clf__min_weight_fraction_leaf': [x/10000 for x in range(3, 19, 3)]}]

        # initiate hyperparameter search
        search(data,
               pipe,
               param_grid,
               "Results/rf_min_weight_prova.csv",
               splits=3,
               size=300000,
               gap=300000)

    if FOREST is True:

        # load grid search results for best values of min_weigt_fraction_leaf
        data = pd.read_csv("Results/rf_min_weight.csv")
        data1 = pd.read_csv("Results/rf_min_weight_1.csv")
        data2 = pd.read_csv("Results/rf_min_weight_2.csv")

        # concatenate dataframes
        df = pd.concat([data, data1, data2])

        # load grid search results from ExtraTree algorithm
        df_extra = pd.read_csv("Results/rf_extra_min_weight.csv")

        # add a new column to the dataframes indicating the algorithm used
        df["classifier"] = df["mean_fit_time"]*0
        df["classifier"].replace(0, "Random Forest", inplace=True)

        df_extra["classifier"] = df_extra["mean_fit_time"]*0
        df_extra["classifier"].replace(0, "Extra Trees", inplace=True)

        all_data = pd.concat([df, df_extra])

        # new dataframe withouth the overfitted models
        best_data = all_data[all_data["mean_train_score"] < 0.60]
        # print hyperparamters of the 3 best performing models
        print("hyperparameters for the 3 best performig models")
        print(best_data.sort_values(by="mean_test_score", ascending=False)[
              ["param_clf__n_estimators", "param_clf__min_weight_fraction_leaf", "classifier"]].head(3))
        print("")

        all_data_1 = all_data.copy()
        all_data_1["f1 score"] = all_data_1["mean_train_score"]
        all_data_1["type"] = all_data_1["f1 score"]*0
        all_data_1["type"].replace(0, "train", inplace=True)

        all_data["f1 score"] = all_data["mean_test_score"]
        all_data["type"] = all_data["f1 score"]*0
        all_data["type"].replace(0, "test", inplace=True)

        new_data = pd.concat([all_data, all_data_1])

        # plot the results
        line = sns.lineplot(data=new_data,
                            y="f1 score",
                            hue="classifier",
                            x="param_clf__min_weight_fraction_leaf",
                            style="type",
                            markers=True)

        plt.grid()  # add grid
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # add legend
        plt.show()

        # repeat the same process for  max depth
        data = pd.read_csv("Results/rf_max_depth.csv")
        data1 = pd.read_csv("Results/rf_max_depth_1.csv")

        df = pd.concat([data, data1])

        df_extra = pd.read_csv("Results/rf_extra_max_depth.csv")

        df["classifier"] = df["mean_fit_time"]*0
        df["classifier"].replace(0, "Random Forest", inplace=True)

        df_extra["classifier"] = df_extra["mean_fit_time"]*0
        df_extra["classifier"].replace(0, "Extra Trees", inplace=True)

        all_data = pd.concat([df, df_extra])

        best_data = all_data[all_data["mean_train_score"] < 0.60]

        print("hyperparameters for the 3 best performig models")
        print(best_data.sort_values(by="mean_test_score", ascending=False)[
              ["param_clf__n_estimators", "param_clf__max_depth", "classifier"]].head(3))
        print("")

        all_data_1 = all_data.copy()
        all_data_1["f1 score"] = all_data_1["mean_train_score"]
        all_data_1["type"] = all_data_1["f1 score"]*0
        all_data_1["type"].replace(0, "train", inplace=True)

        all_data["f1 score"] = all_data["mean_test_score"]
        all_data["type"] = all_data["f1 score"]*0
        all_data["type"].replace(0, "test", inplace=True)

        new_data = pd.concat([all_data, all_data_1])

        line = sns.lineplot(data=new_data,
                            y="f1 score",
                            hue="classifier",
                            x="param_clf__max_depth",
                            style="type",
                            markers=True)

        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        # load random search results for the random forest algorithm
        search = pd.read_csv('Results/rf_random_search.csv')
        search_1 = pd.read_csv('Results/rf_random_search_1.csv')
        search_2 = pd.read_csv('Results/rf_random_search_2.csv')

        all_search = pd.concat([search, search_1, search_2])

        # add a new column to the dataframes indicating the algorithm used
        all_search["classifier"] = all_search["mean_fit_time"]*0
        all_search["classifier"].replace(0, "Random Forest", inplace=True)

        # load random search results for the extra tree algorithm
        search_extra = pd.read_csv('Results/rf_random_extra.csv')
        search_extra_1 = pd.read_csv('Results/rf_random_extra_1.csv')

        all_search_extra = pd.concat([search_extra, search_extra_1])

        # add a new column to the dataframes indicating the algorithm used
        all_search_extra["classifier"] = all_search_extra["mean_fit_time"]*0
        all_search_extra["classifier"].replace(0, "Extra Trees", inplace=True)

        # concatenate the results of the two algorithms
        data_search = pd.concat([all_search, all_search_extra])

        # change some labels
        labels_to_change = ["SelectKBest(k=20)", "SelectKBest(k=30)", "SelectKBest(k=40)"]
        data_search["param_reduce_dim"].replace(labels_to_change, "SelectKBest()", inplace=True)
        data_search["param_reduce_dim"].replace("PCA(n_components=0.93)", "PCA()", inplace=True)

        # plot a scatterplot of the random search results
        scatter = sns.scatterplot(data=data_search,
                                  x="mean_fit_time",
                                  y="mean_test_score",
                                  style="classifier",
                                  hue="param_reduce_dim")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # show legend
        plt.show()

    if NAIVE is True:

        # load first grid search results
        data = pd.read_csv('Results/naive_outliers.csv')

        # delete some duplicate rows from dataframe
        a = range(16, 24)
        data = data.drop(a, axis=0)
        data = data.drop([27, 26], axis=0)

        # load second grid search result
        data1 = pd.read_csv('Results/naive_nice_removed.csv')
        data1["param_miss_value__strategy"] = 'removed'

        # merge the two dataframe
        df = pd.concat([data, data1])

        # replace some labels
        df["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest()", inplace=True)

        # plot a scatterlot of the search results
        g = sns.scatterplot(y="mean_test_score",
                            x="mean_fit_time",
                            style="param_miss_value__strategy",
                            hue="param_reduce_dim",
                            legend="brief",
                            data=df)

        g.set_xlabel("Computational time [sec]")
        g.set_ylabel("Mean f1 score on test set")
        g.set_xscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

        # load model results for the bagged naive bayes with 25 base estimators
        models = {}
        for i in range(1, 4):
            NAME = "data_25_{}".format(i)
            models[NAME] = pd.read_csv('Results/bagging_naive_25_{}.csv'.format(i))

        # merge the dataframes
        df_25 = pd.concat([models[name] for name in models])

        # replace some labels
        df_25["param_reduce_dim"].replace("PCA(n_components=0.91)", "PCA()", inplace=True)
        df_25["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest()", inplace=True)
        df_25["param_reduce_dim"].replace("passthrough", "None", inplace=True)

        # plot a scatterlot of the search results
        g_25 = sns.scatterplot(x="mean_fit_time",
                               y="mean_test_score",
                               hue="param_reduce_dim",
                               style="param_clf__max_features",
                               legend="brief",
                               data=df_25)

        g_25.set_xlabel("Computational time [sec]")
        g_25.set_ylabel("Mean f1 score on test set")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

        # load model results for the bagged naive bayes with 25 base estimators
        models = {}
        for i in range(1, 6):
            NAME = "data_50_{}".format(i)
            models[NAME] = pd.read_csv('Results/bagging_naive_50_{}.csv'.format(i))

            # before merging we need to make sure the dataframes have the same columns
            if "param_clf__max_features" not in models[NAME].columns:
                models[NAME]["param_clf__max_features"] = models[NAME]["mean_fit_time"] * 0 + 1.

            if "param_miss_value__strategy" not in models[NAME].columns:
                models[NAME]["param_miss_value__strategy"] = models[NAME]["mean_fit_time"] * 0
                models[NAME]["param_miss_value__strategy"].replace(0, "None", inplace=True)

        # merge the dataframes
        df_50 = pd.concat([models[name] for name in models])

        # replace some labels
        df_50["param_reduce_dim"].replace("PCA(n_components=0.91)", "PCA()", inplace=True)
        df_50["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest()", inplace=True)
        df_50["param_reduce_dim"].replace("passthrough", "None", inplace=True)

        # plot a scatterlot of the search results
        g_50 = sns.scatterplot(x="mean_fit_time",
                               y="mean_test_score",
                               hue="param_reduce_dim",
                               style="param_clf__max_features",
                               legend="brief",
                               data=df_50)

        g_50.set_xlabel("Computational time [sec]")
        g_50.set_ylabel("Mean f1 score on test set")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

        df["counts"] = df["mean_fit_time"]
        for value in df["param_reduce_dim"].unique():
            NUM = 1
            for i in range(df.shape[0]):
                if df["param_reduce_dim"].iat[i] == value:
                    df["counts"].iat[i] = NUM
                    NUM = NUM+1

        df["param_clf__max_features"] = df["mean_fit_time"] * 0

        # add a new colum to the dataframes indicating the number of estimators
        df["num_estimators"] = df["mean_fit_time"] * 0 + 1
        df_25["num_estimators"] = df_25["mean_fit_time"]*0 + 25
        df_50["num_estimators"] = df_50["mean_fit_time"]*0 + 50

        # drop the results for the MDI feature selection tecnique
        # because of the poor results
        df.drop(range(28, 34), axis=0, inplace=True)

        # merge dataframes
        all_data = pd.concat([df, df_25, df_50])

        # change some labels
        all_data["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest", inplace=True)
        all_data["param_reduce_dim"].replace("PCA(n_components=0.91)", "PCA()", inplace=True)
        all_data["param_reduce_dim"].replace("MDIFeatureSelection()", "MDI", inplace=True)
        all_data["param_reduce_dim"].replace("passthrough", "None", inplace=True)
        all_data["param_miss_value__strategy"].replace("None", "removed", inplace=True)
        all_data["param_reduce_dim"].value_counts()

        # inizialize 2x2 subplot
        fig, axes = plt.subplots(2, 2)

        # plot 4 barplot that compares the performances of the naive bayes models
        sns.barplot(ax=axes[0, 0],
                    x="param_reduce_dim",
                    y="mean_test_score",
                    hue="num_estimators",
                    data=all_data)

        sns.barplot(ax=axes[0, 1],
                    x="param_miss_value__strategy",
                    y="mean_test_score",
                    hue="num_estimators",
                    data=all_data)

        sns.barplot(ax=axes[1, 0],
                    x="param_reduce_dim",
                    y="mean_fit_time",
                    hue="num_estimators",
                    data=all_data)

        sns.barplot(ax=axes[1, 1],
                    x="param_miss_value__strategy",
                    y="mean_fit_time",
                    hue="num_estimators",
                    data=all_data)

        # we don't want each subplot to have a legend so we removed the legend
        # from 3 of them
        axes[1, 1].legend().set_visible(False)
        axes[0, 0].legend().set_visible(False)
        axes[1, 0].legend().set_visible(False)

        # aesthetic changes
        axes[0, 0].set_ylabel("Mean f1 on test")
        axes[1, 0].set_ylabel("Mean fit time [sec]")
        axes[0, 1].set_ylabel("")
        axes[1, 1].set_ylabel("")
        axes[0, 0].set_xlabel("")
        axes[1, 0].set_xlabel("")
        axes[0, 1].set_xlabel("")
        axes[1, 1].set_xlabel("")
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
>>>>>>> d59d53f5651ea14a8f4710139fe10126cfcd2135
