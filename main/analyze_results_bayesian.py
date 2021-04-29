"""
This module is used to visualize the results of the grid searches for the naive
bayes classifier.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=="__main__":
    # load first grid search results
    data = pd.read_csv('Results/naive_outliers.csv')

    # delete some duplicate rows from dataframe
    a = range(16, 24)
    data = data.drop(a, axis=0)
    data = data.drop([27, 26], axis=0)

    # load second grid search result
    data1 = pd.read_csv('Results/naive_nice_removed.csv')
    data1["param_miss_value__strategy"] = 'removed'

    # merge the two dataframe
    df = pd.concat([data, data1])

    # replace some labels
    df["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest()", inplace=True)

    # plot a scatterlot of the search results
    g = sns.scatterplot(y="mean_test_score",
                        x="mean_fit_time",
                        style="param_miss_value__strategy",
                        hue="param_reduce_dim",
                        legend="brief",
                        data=df)

    g.set_xlabel("Computational time [sec]")
    g.set_ylabel("Mean f1 score on test set")
    g.set_xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

    # load model results for the bagged naive bayes with 25 base estimators
    models = {}
    for i in range(1, 5):
        NAME = "data_25_{}".format(i)
        models[NAME] = pd.read_csv('Results/bagging_naive_25_{}.csv'.format(i))

        if "param_miss_value__strategy" not in models[NAME].columns:
            models[NAME]["param_miss_value__strategy"] = models[NAME]["mean_fit_time"] * 0
            models[NAME]["param_miss_value__strategy"].replace(0, "None", inplace=True)

    # merge the dataframes
    df_25 = pd.concat([models[name] for name in models])

    # replace some labels
    df_25["param_reduce_dim"].replace("PCA(n_components=0.91)", "PCA()", inplace=True)
    df_25["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest()", inplace=True)
    df_25["param_reduce_dim"].replace("passthrough", "None", inplace=True)

    # plot a scatterlot of the search results
    g_25 = sns.scatterplot(x="mean_fit_time",
                           y="mean_test_score",
                           hue="param_reduce_dim",
                           style="param_clf__max_features",
                           legend="brief",
                           data=df_25)

    g_25.set_xlabel("Computational time [sec]")
    g_25.set_ylabel("Mean f1 score on test set")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

    # load model results for the bagged naive bayes with 25 base estimators
    models = {}
    for i in range(1, 6):
        NAME = "data_50_{}".format(i)
        models[NAME] = pd.read_csv('Results/bagging_naive_50_{}.csv'.format(i))

        # before merging we need to make sure the dataframes have the same columns
        if "param_clf__max_features" not in models[NAME].columns:
            models[NAME]["param_clf__max_features"] = models[NAME]["mean_fit_time"] * 0 + 1.

        if "param_miss_value__strategy" not in models[NAME].columns:
            models[NAME]["param_miss_value__strategy"] = models[NAME]["mean_fit_time"] * 0
            models[NAME]["param_miss_value__strategy"].replace(0, "None", inplace=True)

    # merge the dataframes
    df_50 = pd.concat([models[name] for name in models])

    # replace some labels
    df_50["param_reduce_dim"].replace("PCA(n_components=0.91)", "PCA()", inplace=True)
    df_50["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest()", inplace=True)
    df_50["param_reduce_dim"].replace("passthrough", "None", inplace=True)

    # plot a scatterlot of the search results
    g_50 = sns.scatterplot(x="mean_fit_time",
                           y="mean_test_score",
                           hue="param_reduce_dim",
                           style="param_clf__max_features",
                           legend="brief",
                           data=df_50)

    g_50.set_xlabel("Computational time [sec]")
    g_50.set_ylabel("Mean f1 score on test set")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

    df["counts"] = df["mean_fit_time"]
    for value in df["param_reduce_dim"].unique():
        NUM = 1
        for i in range(df.shape[0]):
            if df["param_reduce_dim"].iat[i] == value:
                df["counts"].iat[i] = NUM
                NUM = NUM+1

    df["param_clf__max_features"] = df["mean_fit_time"] * 0

    # add a new colum to the dataframes indicating the number of estimators
    df["num_estimators"] = df["mean_fit_time"] * 0 + 1
    df_25["num_estimators"] = df_25["mean_fit_time"]*0 + 25
    df_50["num_estimators"] = df_50["mean_fit_time"]*0 + 50

    # drop the results for the MDI feature selection tecnique
    # because of the poor results
    df.drop(range(28, 34), axis=0, inplace=True)

    # merge dataframes
    all_data = pd.concat([df, df_25, df_50])

    # change some labels
    all_data["param_reduce_dim"].replace("SelectKBest(k=20)", "SelectKBest", inplace=True)
    all_data["param_reduce_dim"].replace("PCA(n_components=0.91)", "PCA()", inplace=True)
    all_data["param_reduce_dim"].replace("MDIFeatureSelection()", "MDI", inplace=True)
    all_data["param_reduce_dim"].replace("passthrough", "None", inplace=True)
    all_data["param_miss_value__strategy"].replace("None", "removed", inplace=True)
    all_data["param_reduce_dim"].value_counts()

    # inizialize 2x2 subplot
    fig, axes = plt.subplots(2, 2)

    # plot 4 barplot that compares the performances of the naive bayes models
    sns.barplot(ax=axes[0, 0],
                x="param_reduce_dim",
                y="mean_test_score",
                hue="num_estimators",
                data=all_data)

    sns.barplot(ax=axes[0, 1],
                x="param_miss_value__strategy",
                y="mean_test_score",
                hue="num_estimators",
                data=all_data)

    sns.barplot(ax=axes[1, 0],
                x="param_reduce_dim",
                y="mean_fit_time",
                hue="num_estimators",
                data=all_data)

    sns.barplot(ax=axes[1, 1],
                x="param_miss_value__strategy",
                y="mean_fit_time",
                hue="num_estimators",
                data=all_data)

    # we don't want each subplot to have a legend so we removed the legend
    # from 3 of them
    axes[1, 1].legend().set_visible(False)
    axes[0, 0].legend().set_visible(False)
    axes[1, 0].legend().set_visible(False)

    # aesthetic changes
    axes[0, 0].set_ylabel("Mean f1 on test")
    axes[1, 0].set_ylabel("Mean fit time [sec]")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].set_xlabel("")
    axes[1, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    axes[1, 1].set_xlabel("")
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    print("hyperparameters for the 3 best performig models")
    print(all_data.sort_values(by="mean_test_score", ascending=False)[
          ["num_estimators","param_clf__max_features",
          "param_reduce_dim__k","param_miss_value__strategy"]].head(3))
    print("")
