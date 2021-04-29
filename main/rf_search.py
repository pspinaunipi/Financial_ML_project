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
