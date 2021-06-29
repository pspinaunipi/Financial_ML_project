"""
This module is used for visualization and analysis of the results about Adaboost
hyperparamters optimization, realized by 50 iteration with GridSearchCV.
We find out the hyperparamters for the 3 best models and we compare the performances
in term of fit time and f1 score for different features selection methods.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=="__main__":
    #read the dataframe for all the searches
    search = pd.read_csv('Results/risultati_adaboost/all_searches.csv')
    labels_to_change1 = ["SelectKBest(k=20)", "SelectKBest(k=30)", "SelectKBest(k=50)"]
    labels_to_change2 = ["PCA(n_components=0.91)","PCA(n_components=0.93)","PCA(n_components=0.95)"]
    search["param_reduce_dim"].replace(labels_to_change1, "SelectKBest()", inplace=True)
    search["param_reduce_dim"].replace(labels_to_change2, "PCA()", inplace=True)

    # scatterplot to compare performances about different feature selection
    # methods and number of estimators
    scatter = sns.scatterplot(data=search,
                                  x="mean_fit_time",
                                  y="mean_test_score",
                                  hue="param_reduce_dim",
                                  style="param_clf__n_estimators")

    plt.legend(bbox_to_anchor=(1.05, 1))  # show legend
    plt.xlabel("Computational time [sec]",fontsize=18)
    plt.ylabel("Mean f1 score on test set",fontsize=18)
    plt.xscale("log")
    plt.legend()
    plt.show()


    #What are the best hyperparamters after our searches?
    print("The hyperparameters for the 3 best performig models are:\n")
    print(search.sort_values(by="mean_test_score", ascending=False)[
          ["param_clf__n_estimators",
            "mean_test_score"]].head(3))
