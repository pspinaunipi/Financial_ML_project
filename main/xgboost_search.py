"""
This is the main module we used to fit the XGBoost models. After defining a pipeline and a
param grid, a RandomizedSearch with CV is initialized and all the important informations
about the search are saved in a location choosen by the user.
"""
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from initial_import import import_training_set


if __name__ == '__main__':
    # load dataset
    data = import_training_set(fast_pc = True)
    data.dropna(inplace=True)
    X = data.drop(["action"], axis=1)
    y = np.ravel(data.loc[:, ["action"]])
    y = np.array(y)

    #set up cv
    cv = TimeSeriesSplit(n_splits=3,test_size=400000,gap=200000)
    # set up classifier and pipeline
    xgb = XGBClassifier (objective='binary:logistic',
                         eval_metric="logloss",
                         nthread=-1,
                         tree_method="hist",
                         use_label_encoder=False)

    #prepare param_grid
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('reduce_dim', 'passthrough'),
                     ('clf', xgb)])

    # set up param grid
    param_grid = {
        'clf__n_estimators':[x for x in range (50,100,5)],
        'clf__min_child_weight': [x for x in range(1,10)],
        'clf__gamma': [x/10 for x in range(5,50,5)],
        'clf__subsample': [x/100 for x in range(6,10)],
        'clf__colsample_bytree': [x/100 for x in range(6,10)],
        'clf__eta':[x/1000 for x in range(3,50)],
        'clf__reg_alpha': [10**(-x/10) for x in range(0,40,5)],
        'clf__reg_lambda': [10**(-x/10) for x in range(0,40,5)],
        'clf__max_depth':[x for x in range(3,9)]
        }

    #set up the search
    random_search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=1,
                                   scoring='f1', n_jobs=-1, cv=cv, verbose=2, random_state=1001,return_train_score=True)


    # fit the search
    random_search.fit(X,y)

    #save sorted results
    df = pd.DataFrame(random_search.cv_results_)
    df= df.sort_values(by="mean_test_score", ascending=False)
    df.to_csv('Results/risultati_xgboost/xgboost_prova1.csv')
    del X,y
    gc.collect()
