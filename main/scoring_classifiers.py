"""
This module is
"""
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,precision_score,recall_score,make_scorer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from initial_import import import_training_set
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest

def compute_scores(train_data,lst_pipes,cv):
    """
    This function is useful to compare the performance of different classifiers according
    to the following scoring metrics: precision,recall,f1 score, roc auc, accuracy.
    A CV is carried out and then all the informations for each split are saved into a
    pandas DataFrame.

    Parameters
    ----------
    train_data: DataFrame
        The training dataset
    lst_pipe: list of Pipeline
        Each element of this list is a different Pipeline
    cv: CV
        The cross validation parameters

    Yields
    ------
    df: DataFrame
    A pandas Dataframe containing the result of the cross validation.

    """
    # separate class label from the rest of the dataset
    X = train_data.drop(["action"], axis=1)
    y = np.ravel(train_data.loc[:, ["action"]])
    y = np.array(y)
    results = []
    scores = ["f1","accuracy","roc_auc","precision","recall"]
    # start cross validation for each pipeline
    for pipe in (lst_pipes):
        cv_result = cross_validate(pipe,X,y,cv=cv,scoring=scores,return_estimator=True,
                                    return_train_score=True,verbose=1)
        #save the results as pandas DataFrame
        results.append(pd.DataFrame(cv_result))
    # merge the results DataFrames into a single one
    df = pd.concat([result for result in results])
    return df

if __name__=="__main__":
    '''
    # load dataset
    data = import_training_set(fast_pc=True)
    data.dropna(inplace=True)
    #define pipelines
    bagging_1 = BaggingClassifier(base_estimator=GaussianNB(),
                                  n_estimators=50,
                                  max_samples=0.25,
                                  max_features=0.33,
                                  verbose=1,
                                  n_jobs=3,
                                  bootstrap=True)

    bagging_2 = BaggingClassifier(base_estimator=GaussianNB(),
                                  n_estimators=50,
                                  max_samples=0.25,
                                  max_features=0.33,
                                  verbose=1,
                                  n_jobs=3,
                                  bootstrap=True)

    bagging_3 = BaggingClassifier(base_estimator=GaussianNB(),
                                  n_estimators=25,
                                  max_samples=0.25,
                                  max_features=0.33,
                                  verbose=1,
                                  n_jobs=3,
                                  bootstrap=True)



    pipe_1 = Pipeline([('miss_value', SimpleImputer(strategy="mean")),
                     ('scaler', StandardScaler()),
                     ('reduce_dim', SelectKBest(k=20)),
                     ('clf', bagging_1)])
    pipe_2 = Pipeline([('miss_value', SimpleImputer(strategy="constant")),
                     ('scaler', StandardScaler()),
                     ('reduce_dim', SelectKBest(k=20)),
                     ('clf', bagging_2)])
    pipe_3 = Pipeline([('miss_value', SimpleImputer(strategy="mean")),
                     ('scaler', StandardScaler()),
                     ('reduce_dim', SelectKBest(k=20)),
                     ('clf', bagging_3)])

    #define cross_validation
    cv = TimeSeriesSplit(n_splits=10,test_size=100000,gap=100000)
    list_pipes = [pipe_1,pipe_2,pipe_3]
    #compute the calssifier scores
    scores = compute_scores(data,list_pipes,cv)
    scores.to_csv("Results/best_3_bayes.csv")


    '''
    pd.set_option('display.max_colwidth', None)
    data = pd.read_csv("Results/best_3_forests.csv",index_col=0)
    final_result= data.groupby(by="estimator").mean()
    print(final_result.to_string(index=False))
