"""
This module recap and analyzes the results of our searches for the hyperparamters
optimization on XGBoost. We realized a lineplot to verify the goodness of max
depth as early stopping condition, based on pre tuning searches only on 3 principal
hyperparameters. Then we see what are the most influencing hyperparamters in terms
of fit time and f1 score, computing the relative correlations.
Finally we visualize in a scatterplot the performances obtained varying subsample
and colsample_bytree values.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  initial_import import import_training_set, compute_action
import xgboost as xgb
import numpy as np

if __name__=="__main__":

    #read all pre searches
    data_pre=pd.read_csv("Results/risultati_xgboost/pre_searches.csv")
    # print hyperparamters of the 3 best performing models
    print("These are the three best performig models:")
    print(data.sort_values(by="mean_test_score", ascending=False)[
          ["param_clf__n_estimators", "param_clf__min_child_weight",
           "param_clf__gamma","param_clf__max_depth",
           "mean_test_f1"]].head(4))
    print("")


    #Verifing early stopping condition max depth
    data_1 = data_pre.copy()
    data_1["F1 score"] = data_1["mean_train_f1"]
    data_1["type"] = data_1["F1 score"]*0
    data_1["type"].replace(0, "train", inplace=True)

    data_pre["F1 score"] = data_pre["mean_test_f1"]
    data_pre["type"] = data_pre["F1 score"]*0
    data_pre["type"].replace(0, "test", inplace=True)

    new_data = pd.concat([data_pre,data_1])
    # plot the results
    line = sns.lineplot(data=new_data,
                        y="F1 score",
                        x="param_clf__colsample_bytree",
                        hue="type",
                        markers=True)

    plt.title("XGBoost colsample_bytree")
    plt.grid()  # add grid
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # add legend
    plt.show()



    #scatter plot to evaluate colsample_by tree vs subsample
    search = pd.read_csv('Results/risultati_xgboost/all_searches.csv')
    #define two class boundary for colsample_bytree values
    search["param_clf__colsample_bytree"][search["param_clf__colsample_bytree"]< 0.7]=0.7
    search["param_clf__colsample_bytree"][search["param_clf__colsample_bytree"]> 0.7]=1
    #define two class boundary for subsample values
    search["param_clf__subsample"][search["param_clf__subsample"]< 0.4]=0.4
    search["param_clf__subsample"][search["param_clf__subsample"]> 0.4]=0.8



    # plot a scatterplot to visualize how to change the performances varying
    # subsample and colsample_bytree values
    scatter = sns.scatterplot(data=search,
                                     x="mean_fit_time",
                                     y="mean_test_f1",
                                     legend="brief",
                                     hue="param_clf__subsample",
                                     palette="bright",
                                     style="param_clf__colsample_bytree")

    plt.legend(bbox_to_anchor=(1.05, 1))  # show legend
    plt.xlabel("Computational time [sec]",fontsize=18)
    plt.ylabel("Mean fit score on test set",fontsize=18)
    plt.xscale("log")
    plt.legend()
    plt.show()


    #plot correlation matrix between hyperparamters and fit time & f1 score
    corr1=data[["param_clf__gamma","param_clf__colsample_bytree","param_clf__max_depth",
             "param_clf__subsample","param_clf__min_child_weight","param_clf__n_estimators",
             "param_clf__eta"]].corrwith(data["mean_test_f1"])
    corr2=data[["param_clf__gamma","param_clf__colsample_bytree","param_clf__max_depth",
             "param_clf__subsample","param_clf__min_child_weight","param_clf__n_estimators",
             "param_clf__eta"]].corrwith(data["mean_fit_time"])

    corr = pd.concat([corr1,corr2],axis=1)
    corr=corr.rename(columns={'0':'mean_test_f1','1': 'mean_fit_time'},inplace=False)
    plt.subplots(figsize=(10, 5))
    sns.heatmap(corr,annot=True)
    plt.show()
