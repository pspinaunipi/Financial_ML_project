import time
import pickle
import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from initial_import import import_training_set
from splitting import split_data
import feature_selection
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from statistics import mean, stdev


def time_cv(clf, n_folds):

    folds = TimeSeriesSplit(n_splits=n_folds,
                            test_size=int(2e5),
                            gap=int(2e5))

    # create model
    splits = folds.split(X, y)

    score_cv = []  # empty list will contain accuracy score of each split

    # start cross validation
    for fold_n, (train_index, val_index) in enumerate(splits):
        print('Fold: {}'.format(fold_n+1))

        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]

        # fit model and compute score
        eval_set = [(X_train, y_train), (X_val, y_val)]
        clf.fit(X_train,
                y_train,
                eval_metric="logloss",
                early_stopping_rounds=25,
                eval_set=eval_set,
                verbose=False)  # fit XGB
        # make predictions for validation data
        y_pred = clf.predict(X_val)
        predictions = [round(value) for value in y_pred]
        accuracy = roc_auc_score(y_val, predictions)
        score_cv.append(accuracy)

        # delete train and validation set to save memory
        del X_train, X_val, y_train, y_val
        gc.collect()

    return score_cv


def acc_model(params):
    """
    This function computes the AUC score of the model on the validation set

    Parameters
    ----------
    params: **kwargs
        paramters for the RF model

    Yields
    ------
    accuracy: float
        AUC score
    """
    clf = XGBClassifier(**params)
    list_auc = time_cv(clf, 5)
    accuracy = mean(list_auc)
    std = stdev(list_auc)

    return accuracy, std


def func(params):
    """
    This is the functon we want to minimize during the tuning process

    Parameters
    ----------
    params: **kwargs
        paramters for the RF model

    Yields
    ------
    loss: float
        the parameter we want to minimize (-AUC score)
    status:
        this parameter reports if an error occured while the model is fitting
    train_time:
        time fit the model and compute the AUC score
    """
    begin = time.time()
    acc, std = acc_model(params)
    print(params)
    print("Mean AUC {:.5f} \nStd {:.5f}".format(acc, std))
    end = (time.time() - begin) / 60
    return {'loss': -acc, 'status': STATUS_OK, 'train_time': end, 'std': std}


if __name__ == "__main__":

    CORRELATION = False
    NEW_START = True
    SKIP_85_DAYS = True
    NO_0_WEIGHT = False
    RF_IMPORTANCE = False
    HYPERPARAMETER_SEARCH = True
    ITERATION = 1000

    data = import_training_set()  # import training set

    # skip first 85 days because of change of JaneStreetMkt trading critieria
    if SKIP_85_DAYS is True:
        data = data[data["date"] > 85]
        data["date"] = data["date"]-85

    # skip 0 weight transaction
    if NO_0_WEIGHT is True:
        data = data[data["weight"] != 0]

    # Remove feature based on correlation
    if CORRELATION is True:
        useless = feature_selection.main(0.93)
        data = data.drop(useless, axis=1)

    # remove features based on MSI feature importance
    if RF_IMPORTANCE is True:
        redunant_feat = np.loadtxt("Results/deleted_feat_skip85.csv", dtype="str")
        data = data.drop(redunant_feat, axis=1)

    if HYPERPARAMETER_SEARCH is True:
        # divide dataset into test training and validation set
        X, y, X_test, y_test = split_data(data)

        # define search_space
        search_space = {
            "n_estimators": 10000,
            "eta": hp.uniform("eta", 0, 0.25),
            "gamma": hp.choice("gamma", range(1, 20)),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "max_depth": hp.choice("max_depth", range(5, 35)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 10, 1)),
            "tree_method": hp.choice("tree_method",  ["hist", "approx"]),
            "n_jobs": -1,
            "label_encoder": False
        }

        # if newstart is true we start optimization from scratch
        if NEW_START is True:
            SEARCHES = 10  # number of searches at the end of a cyle
            start = time.time()  # used to get computational time

            # this will contains all the informations about the search process
            trials = Trials()

            # search hyperparameters
            best = fmin(func, search_space, algo=tpe.suggest, max_evals=10, trials=trials)

            # nice prints
            print('best:')
            print(best)
            finish = (time.time()-start)/60  # time to fit model in minutes
            print("Time to search hyperparameters {:.2f} min".format(finish))

            # The trials database now contains 10 entries, it can be saved/reloaded with pickle

            pickle.dump(trials, open("Hyperopt/myfile3.p", "wb"))
            # write on a txt file number of trials done
            search_file = open("Hyperopt/searches3.txt", "w")
            search_file.write("{}".format(SEARCHES))
            search_file.close()

        # now we can continue the optimization process from when we stopped
        CONTINUE = True

        while CONTINUE is True:

            # load previous results
            trials = pickle.load(open("Hyperopt/myfile3.p", "rb"))
            start = time.time()  # used to get computational time

            # load total numer of evaluation already done
            textfile = open("Hyperopt/searches3.txt", "r")
            SEARCHES = int(textfile.read())
            textfile.close()

            SEARCHES = SEARCHES+10  # at the end of the cycle we will have ten more evaluations

            # searh hyperparameters
            best = fmin(func, search_space, algo=tpe.suggest, max_evals=SEARCHES, trials=trials)
            finish = (time.time()-start)/60  # time to fit model in minutes

            # nice prints
            print("Time to search hyperparameters {} min".format(finish))
            print('best:')
            print(best)

            # The trials database now contains 10 +searches entries
            # it can be saved/reloaded with pickle
            pickle.dump(trials, open("Hyperopt/myfile3.p", "wb"))

            # save new number of total evaluations
            search_file = open("Hyperopt/searches3.txt", "w")
            search_file.write("{}".format(SEARCHES))
            search_file.close()

            # continue the cycle until we press n
            save = input("Done :) \nDo you want to continue?\ny/n\n")
            if save == "n":
                CONTINUE = False

    else:
        # divide test and training set
        X, y, X_test, y_test = split_data(data)
        print(X.shape)

        # define parameters of cross validation
        N_FOLDS = 5
        folds = TimeSeriesSplit(n_splits=N_FOLDS,
                                test_size=int(2e5),
                                gap=int(2e5))

        # best hyperparameters for the model

        parameters = {'criterion': "gini",
                      'n_estimators': 577,
                      'max_features': "auto",
                      'max_depth': 22,
                      'bootstrap': True,
                      'min_samples_leaf': 6,
                      "max_samples": 0.1259,
                      'verbose': 1,
                      'n_jobs': -1,
                      'random_state': 18}

        # create model
        splits = folds.split(X, y)
        forest = ExtraTreesClassifier(**parameters)
        score_cv = []  # empty list will contain accuracy score of each split
        start = time.time()

        # start cross validation
        for fold_n, (train_index, val_index) in enumerate(splits):
            print('Fold: {}'.format(fold_n+1))

            # for each iteration we define the boundaries of training and validation set
            X_train = X.iloc[train_index[0]:train_index[-1], :]
            X_val = X.iloc[val_index[0]:val_index[-1], :]
            y_train = y[train_index[0]:train_index[-1]]
            y_val = y[val_index[0]:val_index[-1]]

            # fit model and compute score
            forest.fit(X_train, y_train)
            score = forest.score(X_val, y_val)
            score_cv.append(score)
            print(score_cv)

            # delete train and validation set to save memory
            del X_train, X_val, y_train, y_val
            gc.collect()

        finish = (time.time()-start)/60  # time to fit model in minutes

        # nice prints
        print("accuracy score of each iteration is")
        print(score_cv)
        print("Time to fit the RF with {}k cross validation {:.2f} min".format(N_FOLDS, finish))

        # save model
        filename = "RF_6.sav"
        pickle.dump(forest, open(filename, "wb"))

        # compute accuracy on test and training set
        score_test = forest.score(X_test, y_test)
        score_training = forest.score(X, y)
        print("Accuracy on training set is: {} \nAccuracy on test set is: {}".format(
            score_training, score_test))

        # create a dataframe that sumarizes the results
        results = {"score test ": score_test, "score training": score_training,
                   "computational time (min)": finish, "splits": N_FOLDS}
        end_results = pd.DataFrame(results, index=["values"])

        # nice print
        print("a little recap of the results:")
        print(end_results)

        # save results and removed features as csv
        end_results.to_csv("Results/results_cv_subsample{}.csv".format(ITERATION))

        # compute AUC score
        pred_train = forest.predict(X)
        pred_test = forest.predict(X_test)
        auc_train = roc_auc_score(y, pred_train)
        auc_test = roc_auc_score(y_test, pred_test)
        print("AUC on training set is: {} \nAUC on test set is: {}".format(
            auc_train, auc_test))
