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

    print("Best 3 overall models:")
    print(all_search.sort_values(by="mean_test_score", ascending=False)[
          ["param_clf__n_estimators", "classifier","mean_test_score"]].head(3))
    print("")
