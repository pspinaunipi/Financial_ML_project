"""
This is the module we used to make the computations descibed in the chapter about probability.
At the beginning of the main 3 boolean variables can be set.

1)COMPUTE_PROB:
    If set to True a user can set up a pipeline containing a classifier and 5 .csv
    files will be added in the cartella Proba, containing the classification probability
    of each object of 5 different test-sets. At each iteration the classifier is trained on a
    different training set.

2)COMPUTE_ACC:
    If set to True computes the accuracy over different probability thresholds

3)DISTRIBUTION:
    If set to True an image showing the probability distribution of one of the classifier
    used in COMPUTE_PROB is returned
"""

import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


def compute_probabilities(data, pipe, name):
    """
    This function trains a classifier and then computes the class probabilities
    for a test set compose by the last 50 days of the dataset. This function returns a
    .csv file located in the filepath given by the user

    Parameters
    ----------
    data: DataFrame
        The dataset used to fit the model
    pipe: Pipeline
        The pipeline for the classifier
    name: string
        the name used to save the dataframe containig all the informations about the
        search

    Yields
    ------
    A csv file, saved in the location indicated by the filepath


    """

    # divide in train and test set leaving a 50 days gap between them
    max_date = data["date"].max()
    X_train = data[data["date"] < max_date-100].drop(['action'], axis=1)
    y_train = data[data["date"] < max_date-100].loc[:, ['action']].to_numpy()
    y_train = np.ravel(y_train)

    # the test set i composed of the last 50 days of trading
    X_test = data[data["date"] > max_date-50].drop(['action'], axis=1)
    y_test = data[data["date"] > max_date-50].loc[:, ['action']].to_numpy()
    y_test = np.ravel(y_test)

    # fit the model
    pipe.fit(X_train, y_train)

    # compute probability
    probability1 = pipe.predict_proba(X_test)

    # create a dataframe for the probabilities
    prob = pd.DataFrame({'first classifier': probability1[:, 1],
                         'label': y_test})

    # save the dataframe
    prob.to_csv('proba/probabilities_{}.csv'.format(name))


def compute_accuracy(filepath, vmax=None, vmin=500, step=10, label=1):
    """
    This function takes as imput a dataframe created by the function compute_probabilities
    and then it returns a dataframe containig important informations such as
    1) The classification accuracy on the label 1
    2) The ratio of the labels classified as 1
    3) The name of the classifier
    4) The probability value
    5) the function

    Parameters
    ----------
    filepath: str
        The filepath of the dataframe
    vmax: int [0,1000] (default=None)
        The maximum probability threshold considered. If vmax=None it is chosen automatically
        as the minumum between the second highest treshold that returns 0 and the tenth highest threshod
        that returns 1.
    vmin: int [0,1000] (default=500)
        The lowest probability threshold considered
    step: int [0,vmax-vmin] (default=10)
        the probability sample rate
    label: string
        the name of the classifier considered

    Yields
    ------
    results: DataFrame
        A dataframe containing the information described above
    """

    # load dataframe cointaing probabilities
    prob = pd.read_csv(filepath)
    # save number of data on the test set
    size = len(prob)

    if vmax is None:
        # chose vmax as the minumum between the second highest treshold that returns 0
        # and the tenth highest threshold that returns 1
        sorted = prob[prob["label"] == 0].sort_values(by="first classifier", ascending=False)
        sorted1 = prob[prob["label"] == 1].sort_values(by="first classifier", ascending=False)
        vmax1 = int(sorted["first classifier"].iloc[1]*1000)
        vmax2 = int(sorted1["first classifier"].iloc[9]*1000)
        vmax = min(vmax1, vmax2)

        # delete useless datas
        del sorted, sorted1, vmax1, vmax2
        gc.collect()

    lenght = int((vmax-vmin)//step+1)
    acc1 = np.zeros(lenght)
    num1 = np.zeros(lenght)
    iter1 = np.zeros(lenght)
    i = 0

    for thresh in range(vmin, vmax, step):

        iter1[i] = thresh/1000
        # ratio of correctlty classified objects as 1
        yes = prob[prob["first classifier"]
                   > thresh/1000]["label"].value_counts()[1]

        # ratio of wrongly classified objects as 1
        no = prob[prob["first classifier"]
                  > thresh/1000]["label"].value_counts()[0]

        # save accuracy on the label 1 and ratio of the objects classified as 1
        acc1[i] = (yes/(yes+no))
        num1[i] = ((yes+no)/size)

        # next iteration
        i = i+1

    # create a list containig the name of the classifier
    clf1 = [label for x in acc1]

    # create a dataframe with all the useful informations
    results = pd.DataFrame({'accuracy': acc1,
                            'number of label 1': num1,
                            'classifier': clf1,
                            'iteration': iter1,
                            'func': (2*acc1-1)*num1})

    return results


if __name__ == "__main__":

    COMPUTE_PROB = False
    COMPUTE_ACC = False
    DISTRIBUTION = True

    if COMPUTE_PROB is True:

        # import dataset
        data = import_training_set()
        data.fillna(0, inplace=True)
        # select best classifier to compare
        forest = RandomForestClassifier(
            n_estimators=500,
            bootstrap=True,
            max_samples=0.10,
            max_features="auto",
            min_weight_fraction_leaf=0.0013,
            n_jobs=-1)
        bagging_clf = BaggingClassifier(base_estimator=GaussianNB(),
                                        n_estimators=50,
                                        bootstrap=True,
                                        max_samples=0.25,
                                        n_jobs=1,
                                        max_features=0.33)
        bagging = Pipeline([('scaler', StandardScaler()),
                            ('reduce_dim', SelectKBest(k=30)),
                            ('clf', bagging_clf)])
        naive = Pipeline([('scaler', StandardScaler()),
                          ('reduce_dim', SelectKBest(k=30)),
                          ('clf', GaussianNB())])
        # create list of classifier we want to compare
        classifiers = [forest, bagging, naive]
        names = ["forest", "bag_50", "naive"]
        # create a copy of the dataset
        new_data = data.copy()
        # variable used to indicate wich classifier we are considering
        j = 0
        # iterate on each classifier
        for clf in classifiers:
            # iterate 5 times to make sure we have robust results
            for i in range(5):
                max_date = new_data["date"].max()
                label = "{}_{}".format(names[j], i)
                # compute the probabilities
                test_size = compute_probabilities(new_data, clf, label)
                # at each iteration reduce the dataset by 25 days
                new_data = new_data[new_data["date"] < max_date-25]
            # increase j to make sure we are testing the next classifier
            j = j+1
            # delete new_data since it now has 125 less days than the original dataset
            del max_date
            del new_data
            gc.collect()
            # new_data is now equal to the original dataset
            new_data = data.copy()

    if COMPUTE_ACC is True:

        # create a dictionary that will contain the models
        models = {}
        # list containing the names of the models
        # we will have 5 models because we did a 5 split CV before
        names = ["model_0", "model_1", "model_2", "model_3", "model_4"]
        # create a dataframe for each forest and then merge them
        for i, name in enumerate(names):
            clf_name = "proba/probabilities_forest_{}.csv".format(int(i))
            models[name] = compute_accuracy(clf_name, vmin=500, step=2, label=i)
        results = pd.concat([models[name] for name in names])
        # create a dataframe for each naive bayes and then merge them
        for i, name in enumerate(names):
            clf_name = "proba/probabilities_naive_{}.csv".format(int(i))
            models[name] = compute_accuracy(clf_name, vmin=500, step=10, label=i)
            print(name)
        results1 = pd.concat([models[name] for name in names])
        # create 2 subplots 1 for the random forest and 1 for the naive bayes
        fig, axes = plt.subplots(1, 2)
        sns.lineplot(ax=axes[0], data=results, x="iteration", y="func", color="blue", ci=None)
        sns.lineplot(ax=axes[1], data=results1, x="iteration", y="func", color="orange", ci=None)
        plt.xlabel("percentage estimators")
        plt.show()

    if DISTRIBUTION is True:

        data = pd.read_csv("proba/probabilities_forest_0.csv")
        results = compute_accuracy("proba/probabilities_forest_0.csv", step=3)

        mean = data["first classifier"].mean()
        std = data["first classifier"].std()
        print(mean, std)

        g = sns.scatterplot(data=results, y="accuracy", x="iteration", color="black")
        plt.legend()
        # g.set_yscale('log')
        g.set_xlabel("Probability")
        g.set_ylabel("Accuracy")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.show()

        g = sns.histplot(data=data, x="first classifier", color="black")
        plt.show()
