"""
This module is used to visualize the results of the grid searches for the random
forest algorithm.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__=="__main__":

    # load grid search results for best values of min_weigt_fraction_leaf
    data = pd.read_csv("Results/rf_min_weight.csv")
    data1 = pd.read_csv("Results/rf_min_weight_1.csv")
    data2 = pd.read_csv("Results/rf_min_weight_2.csv")
    data4 = pd.read_csv("Results/rf_min_weight_4.csv")

    # concatenate dataframes
    df = pd.concat([data, data1, data2,data4])

    # load grid search results from ExtraTree algorithm
    df_extra_1 = pd.read_csv("Results/rf_extra_min_weight.csv")
    df_extra_2 = pd.read_csv("Results/extra_min_weight_4.csv")
    # concatenate dataframes
    df_extra= pd.concat([df_extra_1,df_extra_2])
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
          ["param_clf__n_estimators", "param_clf__min_weight_fraction_leaf", "classifier","mean_test_score"]].head(3))
    print("")

    all_data_1 = all_data.copy()
    all_data_1["f1 score"] = all_data_1["mean_train_score"]
    all_data_1["type"] = all_data_1["f1 score"]*0
    all_data_1["type"].replace(0, "train", inplace=True)

    all_data["f1 score"] = all_data["mean_test_score"]
    all_data["type"] = all_data["f1 score"]*0
    all_data["type"].replace(0, "test", inplace=True)

    new_data = pd.concat([all_data, all_data_1])
    plt.figure(figsize=(10,5))
    # plot the results
    line = sns.lineplot(data=new_data,
                        y="f1 score",
                        hue="classifier",
                        x="param_clf__min_weight_fraction_leaf",
                        style="type",
                        markers=True)

    plt.grid()  # add grid
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=11)  # add legend
    plt.xlabel("Min weight fraction leaf",fontsize=14)
    plt.ylabel("Mean F1 score",fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/min_w.png",dpi=300)
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
          ["param_clf__n_estimators", "param_clf__max_depth", "classifier","mean_test_score"]].head(3))
    print("")

    all_data_1 = all_data.copy()
    all_data_1["f1 score"] = all_data_1["mean_train_score"]
    all_data_1["type"] = all_data_1["f1 score"]*0
    all_data_1["type"].replace(0, "train", inplace=True)

    all_data["f1 score"] = all_data["mean_test_score"]
    all_data["type"] = all_data["f1 score"]*0
    all_data["type"].replace(0, "test", inplace=True)

    new_data = pd.concat([all_data, all_data_1])
    plt.figure(figsize=(10,5))
    line = sns.lineplot(data=new_data,
                        y="f1 score",
                        hue="classifier",
                        x="param_clf__max_depth",
                        style="type",
                        markers=True)

    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=11)
    plt.xlabel("Max depth",fontsize=14)
    plt.ylabel("Mean F1 score",fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/max_d.png",dpi=300)
    plt.show()

    # load random search results for the random forest algorithm
    search = pd.read_csv('Results/new_random_forest_1.csv')
    search_1 = pd.read_csv('Results/new_random_forest_2.csv')
    search_2 = pd.read_csv('Results/new_random_forest_3.csv')

    search_1["reduce_dim"] = search_1["mean_fit_time"]*0
    search_1["reduce_dim"] = "PCA"
    search_2["reduce_dim"] = search_1["mean_fit_time"]*0
    search_2["reduce_dim"] = "SelectKBest"
    all_search = pd.concat([search_1, search_2])

    # add a new column to the dataframes indicating the algorithm used
    all_search["classifier"] = all_search["mean_fit_time"]*0
    all_search["classifier"].replace(0, "Random Forest", inplace=True)

    # load random search results for the extra tree algorithm
    search_extra = pd.read_csv('Results/new_extra_trees_1.csv')
    search_extra_1 = pd.read_csv('Results/new_extra_trees_2.csv')
    search_extra_2 = pd.read_csv('Results/new_extra_trees_3.csv')

    search_extra_1["reduce_dim"] = search_extra_1["mean_fit_time"]*0
    search_extra_1["reduce_dim"] = "PCA"
    search_extra_2["reduce_dim"] = search_extra_1["mean_fit_time"]*0
    search_extra_2["reduce_dim"] = "SelectKBest"
    all_search_extra = pd.concat([search_extra_1,search_extra_2])

    # add a new column to the dataframes indicating the algorithm used
    all_search_extra["classifier"] = all_search_extra["mean_fit_time"]*0
    all_search_extra["classifier"].replace(0, "Extra Trees", inplace=True)

    # concatenate the results of the two algorithms
    data_search = pd.concat([all_search, all_search_extra])
    '''
    # change some labels
    labels_to_change = ["SelectKBest(k=20)", "SelectKBest(k=30)", "SelectKBest(k=40)"]
    data_search["param_reduce_dim"].replace(labels_to_change, "SelectKBest()", inplace=True)
    data_search["param_reduce_dim"].replace("PCA(n_components=0.93)", "PCA()", inplace=True)
    '''
    plt.figure(figsize=(10,5))
    # plot a scatterplot of the random search results
    scatter = sns.scatterplot(data=data_search,
                              x="mean_fit_time",
                              y="mean_test_score",
                              style="classifier",
                              hue="reduce_dim")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=11)  # show legend
    plt.xlabel("Fit time [s]",fontsize=14)
    plt.ylabel("Mean F1 score",fontsize=14)
    plt.tight_layout()
    plt.savefig("Figures/random_search.png",dpi=300)
    plt.show()

    print("Best 3 overall models:")
    print(all_search.sort_values(by="mean_test_score", ascending=False)[
          ["param_clf__n_estimators", "classifier","mean_test_score"]].head(3))
    print("")
