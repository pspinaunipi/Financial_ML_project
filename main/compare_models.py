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

    # divide in train and test set
    max_date = data["date"].max()
    X_train = data[data["date"] < max_date-100].drop(['action'], axis=1)
    y_train = data[data["date"] < max_date-100].loc[:, ['action']].to_numpy()
    y_train = np.ravel(y_train)

    X_test = data[data["date"] > max_date-50].drop(['action'], axis=1)
    y_test = data[data["date"] > max_date-50].loc[:, ['action']].to_numpy()
    y_test = np.ravel(y_test)

    # fit 3 best model
    pipe.fit(X_train, y_train)

    # compute probability
    probability1 = pipe.predict_proba(X_test)

    # create a dataframe for the probabilities
    prob = pd.DataFrame({'first classifier': probability1[:, 1],
                         'label': y_test})

    prob.to_csv('proba/probabilities_{}.csv'.format(name))


def compute_accuracy(filepath, vmax=None, vmin=500, step=10):

    # load dataframe cointaing probabilities
    prob = pd.read_csv(filepath)
    # save number of data on the test set
    size = len(prob)

    if vmax is None:
        # get max probability threshold for an incorrect classification
        vmax = int(prob[prob["label"] == 0]["first classifier"].max()*1000)

    lenght = int((vmax-vmin)//step)

    acc1 = np.zeros(lenght)
    num1 = np.zeros(lenght)
    iter1 = np.zeros(lenght)
    i = 0

    for thresh in range(vmin, vmax, step):
        iter1[i] = thresh/1000
        yes = prob[prob["first classifier"]
                   > thresh/1000]["label"].value_counts()[1]
        no = prob[prob["first classifier"]
                  > thresh/1000]["label"].value_counts()[0]

        # save percentage of correct labels and percentage of test set classified as 1
        acc1[i] = (yes/(yes+no))
        num1[i] = ((yes+no)/size)

        # next iteration
        i = i+1

    return acc1, num1, iter1


if __name__ == "__main__":

    COMPUTE_PROB = False
    COMPUTE_ACC = True

    if COMPUTE_PROB is True:

        #import dataset
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

        tree = DecisionTreeClassifier(max_depth=5)

        naive = Pipeline([('scaler', StandardScaler()),
                          ('reduce_dim', SelectKBest(k=30)),
                          ('clf', GaussianNB())])

        # create list of classifier we want to compare
        classifiers = [forest, bagging, tree, naive]
        names = ["forest", "bag_50", "tree", "naive"]

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

                # at each iteration reduc the dataset by 25 days
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

        # choose
        acc1_0, num1_0, iter1_0 = compute_accuracy("proba/probabilities_forest_0.csv",
                                                   vmax=550, vmin=450, step=2)

        acc1_1, num1_1, iter1_1 = compute_accuracy("proba/probabilities_forest_1.csv",
                                                   vmax=550, vmin=450, step=2)
        acc1_2, num1_2, iter1_2 = compute_accuracy("proba/probabilities_forest_2.csv",
                                                   vmax=550, vmin=450, step=2)
        acc1_3, num1_3, iter1_3 = compute_accuracy("proba/probabilities_forest_3.csv",
                                                   vmax=550, vmin=450, step=2)
        acc1_4, num1_4, iter1_4 = compute_accuracy("proba/probabilities_forest_4.csv",
                                                   vmax=550, vmin=450, step=2)

        # compute mean of percentage of correct labels and of percentage of test set classified as 1
        acc = (acc1_0+acc1_1+acc1_2+acc1_3+acc1_4)/5
        num = (num1_0+num1_1+num1_2+num1_3+num1_4)/5

        results = pd.DataFrame({'accuracy': acc,
                                'number of label 1': num,
                                'function': (num*(2*acc-1)),
                                'iteration': iter1_1})

        sns.lineplot(data=results, x="iteration", y="function")

        plt.xlabel("percentage estimators")

        plt.show()
        print("hi")

        print(results.sort_values(by=["function"], ascending=False))
