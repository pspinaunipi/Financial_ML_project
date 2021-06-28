"""
In this module we implemented a user window for the analysis of the competition
database with some usefull functions to plot relevant features, compute the
most important quantities (such as profit) and study the correlation between features.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import initial_import
import seaborn as sns
from matplotlib import cm
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


def statistical_matrix(d_frame: pd.DataFrame):
    """
    This function returns a Pandas dataframe containing useful statistical informations
    about the dataset (mean, median, max , min etc...).
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dataset we want to describe.
    Yields
    ------
    new_matrix: Pandas dataframe
        The dataframe in which we save all the information about our dataset.
    """
    # delete date and date id from columns
    d_frame = d_frame.drop(["date", "ts_id"], axis=1)
    # create a new matrix with all usefull information
    new_matrix = d_frame.describe()
    return new_matrix


def daily_avarage(d_frame: pd.DataFrame):
    """
    This function computes the daily avarage values of each feature.
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dataset on which we compute our daily average quantities.
    Yields
    ------
    day_mean: Pandas dataframe
        The dataframe composed by the daily average values.
    """
    day_mean = d_frame.groupby("date").mean()
    return day_mean


def compute_profit(d_frame: pd.DataFrame):
    """
    This function compute the maximum possible profit in terms of the variables
    recommended in the competition evaluation page.
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dataset in which we want to find the maximum profit
    Yields
    ------
    days: int
        Time range considered in terms of trading days
    val_u:
        Value of the maximum possible utility
    """
    # find the last day of trading
    days = int(d_frame.loc[:, "date"].iat[-1])
    # compute a Pandas series p_i from the original dataset
    p_i = d_frame.loc[:, ["weighted_resp", "date", "action"]]
    p_i["weighted_resp"] = p_i["action"]*p_i["weighted_resp"]
    p_i = p_i.groupby("date").sum()  # sum for each day
    p_i = p_i.loc[:, "weighted_resp"]  # discard other colums
    # compute t and u
    val_t = p_i.sum()/np.sqrt((p_i**2).sum())*np.sqrt(250/days)
    val_u = min(max(val_t, 0), 6)*p_i.sum()
    print("If we take an action every time we have resp > 0.")
    print("The expected utility is {: .3f} after {} days of traiding .\n".format(
        val_u, days))
    return (val_u, days)


def corr_filter(d_frame: pd.DataFrame, bound: float):
    """
    This function analyzes the correlation between the features and identifies
    the highly correlated features.
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dateset on which we evaluate the features correlation.
    bound: float
        This bound identifies our correlation range.
    Yields
    ------
    data_flattened: Pandas series
        The series which contains feature pairings with a correlation >bound or <-bound.
    data_corr: Pandas DataFrame
        Correlation matrix.
    """
    # compute our correlations
    data_corr = d_frame.corr()
    # selct the correlation in the choosen range
    data_filtered = data_corr[((data_corr >= bound) | (
        data_corr <= -bound)) & (data_corr != 1.000)]
    # discard the other values
    data_flattened = data_filtered.unstack().sort_values().drop_duplicates()
    return data_flattened, data_corr


def activity_choice():
    """
    This fuction prints a list of activity on the dataset and return a value
    corresponding to the choosen activity.
    """
    return input("What do you want to do? \n1)Compute statistical matrix \
                \n2)Plot main features over time \n3)Correlation analysis \
                \n4)Plot anonymous features over time \n5)Missing data analysis \
                \n6)Usefull histograms for main features \n7)Usefull histograms for anonymous features \
                \n8)Boxplot for main features\n9)Exit programm\n")


def plot_main_features(data):
    """
    This function is used to plot main features over time in terms of cumultaive
    sum of resp.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # compute daily mean of each feature
    mean_features = daily_avarage(data)
    print("Plotting...\n")
    # plot daily avarage cumulative sums of resps
    plt.figure(figsize=(10, 5))
    plt.title("Cumulative sum of resps", fontsize=20)
    plt.xlabel("Days")
    plt.ylabel("Resp")
    plt.plot(mean_features["resp"].cumsum(), lw=3, label="resp")
    plt.plot(mean_features["resp_1"].cumsum(), lw=3, label="resp_1")
    plt.plot(mean_features["resp_2"].cumsum(), lw=3, label="resp_2")
    plt.plot(mean_features["resp_3"].cumsum(), lw=3, label="resp_3")
    plt.plot(mean_features["resp_4"].cumsum(), lw=3, label="resp_4")
    plt.plot(mean_features["weighted_resp"].sum(), lw=3, label="weighted_resp")
    plt.legend()
    save_oneplot_options("Figures/cumsum_resps.png")


def correlation_analysis(data):
    """
    This function prints higly correlated feature pairings

    """
    '''
    higly_corr_feat = pd.read_csv("Matrices/features_to_remove.csv")
    num_feat = higly_corr_feat.shape[0]
    print("The number of pairings with correlation > 0.90 is {}.\n" .format(num_feat))
    print("Those are:")
    print(higly_corr_feat)

    corr_matrix = pd.read_csv("Matrices/correlation_matrix.csv", index_col=0)
    g = sns.heatmap(corr_matrix, cmap="coolwarm")
    g.set_title("Correlation matrix", fontsize=20)
    plt.show()

    # Plotting most correlated features
    fig, axes = plt.subplots(3, 3)

    fig.suptitle('Scatterplot most correlated features', fontsize=20)
    i = 0
    j = 0
    for k in range(1, 10):
        if k == 1:
            sns.scatterplot(ax=axes[i, j],
                            x=higly_corr_feat.iloc[k, 0],
                            y=higly_corr_feat.iloc[k, 1],
                            hue="action",
                            legend=True,
                            data=data)
        else:
            sns.scatterplot(ax=axes[i, j],
                            x=higly_corr_feat.iloc[k, 0],
                            y=higly_corr_feat.iloc[k, 1],
                            hue="action",
                            legend=False,
                            data=data)
        i = i+1
        if i == 3:
            i = 0
            j = j+1

    plt.show()

    corr_matrix.plot(kind='bar',
                     y=["action"],
                     xticks=[],
                     xlabel="features",
                     ylabel="correlation value")

    plt.suptitle("Correlation with action", fontsize=20)
    plt.show()


    plt.title("Feature 120-129 ove time", fontsize=20)
    plt.xlabel("Days")
    plt.ylabel("Value")
    for k in range(120, 130):
        feat = ["feature_{}".format(k)]
        plt.plot(data.groupby("date").mean()[feat].cumsum(), lw=2, label=feat)

    plt.label()
    plt.show()
    '''
    data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
               "weight", "weighted_resp"], axis=1, inplace=True)
    data.fillna(0, inplace=True)

    X = data.drop(['action'], axis=1)
    y = data.loc[:, ['action']].to_numpy()
    y = np.ravel(y)

    X = StandardScaler().fit_transform(X, y)
    pca = PCA()
    comp = pca.fit(X)

    # We plot a graph to show how the explained variation in the 129 features varies with the number of principal components
    plt.plot(np.cumsum(comp.explained_variance_ratio_))
    plt.title("PCA and variance")
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance')
    sns.despine()
    plt.show()

    pca = PCA(25)
    X = pca.fit_transform(X)
    X = np.column_stack((X, y))
    X_frame = pd.DataFrame.from_records(X)
    new_corr = X_frame.corr()
    print(X_frame)

    g7 = sns.heatmap(new_corr, cmap="Reds")
    g7.set_title("PCA and correlation", fontsize=20)
    plt.show()

    new_corr.plot(kind='bar',
                  y=25,
                  xticks=[],
                  xlabel="features",
                  ylabel="correlation value")

    plt.suptitle("PCA and Correlation with action", fontsize=20)
    plt.show()


def plot_anon_features(data):
    """
    This function is used to plot the anonymous features over time (to see if
    there are some patterns). The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # built matrix containing the anonymous features
    data_anon = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
                           "weight", "weighted_resp", "action", "ts_id"], axis=1)
    names = []  # empty list, it will store the names of the png files
    print("Working on plots of features over time...\n")
    mean_matrix_anon = daily_avarage(data_anon)
    # create 14 images 3x3 containig plot of each anonimous features
    for i in range(14):
        mean_matrix_anon.iloc[:, (9*i):(9*i+9)].plot(subplots=True, layout=(
            3, 3), figsize=(7., 7.))
        plt.subplots_adjust(wspace=0.4)
        plt.suptitle("Anonymous features over time", fontsize=20)
        # storing names of png files in case we want to save them
        names.append('Figures/anonimous_features_over_time{}.png'.format(i))
    save_plots_options(names)


def missing_data_analysis(data, threshold):
    """
    This function is used to identify features with a number of missing values
    over a threshold and build the relative bar plots.
    The user can decide if save or not the figure obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    threshold: float
        Threshold choosen for the selection of features with most missing values.
    """
    '''
    # count number of missing data
    miss_values = data.shape[0]-data.count()
    # select features with missing values over a choosen threshold
    miss_values = miss_values[(miss_values > data.count()*threshold)]
    # plot the relative bar plot
    miss_values.plot(kind="bar", fontsize=10, figsize=(10, 6))
    plt.title("Features with most missing values", fontsize=20)
    plt.subplots_adjust(bottom=0.2)
    save_oneplot_options("Figures/missing_data.png")

    # save indexes of rows with missing data
    missing = data.dropna()
    print("The total number of missing values is {}".format(data.shape[0]-missing.shape[0]))
    missing_index = missing.index.to_numpy()

    del missing
    gc.collect()

    # add a new categorical column to the DataFrame called missing
    # if 0 the corresponding row has no missing missing values
    # if 1 it has missing values
    data["missing"] = data["action"]*0
    for index in missing_index:
        data.at[index, "missing"] = 1

    # plot histogram that visualize the relationship between missing values and action
    g = sns.histplot(x="missing",
                     hue="action",
                     stat="probability",
                     multiple="stack",
                     bins=2,
                     data=data)

    # aestethic stuff
    g.set_title("Missing values and action", fontsize=20)
    g.set_xlabel("Missing values")
    g.set_ylabel("Frequency")

    # save plot
    save_oneplot_options("Figures/missing_data_and_action.png")

    g1 = sns.histplot(hue="missing",
                      x="ts_id",
                      stat="count",
                      multiple="layer",
                      bins=30,
                      data=data)
    plt.show()

    g2 = sns.histplot(hue="missing",
                      x="date",
                      stat="count",
                      multiple="layer",
                      bins=50,
                      data=data)
    plt.show()
    '''
    x_vars = ["feature_7", "feature_8", "feature_17", "feature_18", "feature_27", "feature_28",
              "feature_72", "feature_78", "feature_84", "feature_90",
              "feature_96", "feature_102", "feature_108", "feature_114"]

    #g3 = sns.PairGrid(data, hue="missing", x_vars=x_vars, y_vars=y_vars)
    #g3.map_diag(sns.histplot, color=".3")
    # g3.map_offdiag(sns.scatterplot)
    # g3.add_legend()
    # plt.show()

    plt.title("Missing values over time", fontsize=20)
    plt.xlabel("Transaction")
    plt.ylabel("Value")
    for feature in x_vars:
        plt.plot(data[feature].cumsum(), lw=2, label=feature)
    plt.legend()
    plt.show()
    # delete missinf column from dataset
    #data.drop("missing", axis=1, inplace=True)
    # gc.collect()

    similarity = np.zeros((14, 14))

    miss = data[x_vars].isna()
    for i, element in enumerate(miss.columns):
        for j, value in enumerate(x_vars):
            similarity[i, j] = 1 - cosine(miss[element], miss[value])

    x_lab = ["feat_7", "feat_8", "feat_17", "feat_18", "feat_27", "feat_28",
             "feat_72", "feat_78", "feat_84", "feat_90",
             "feat_96", "feat_102", "feat_108", "feat_114"]

    y_lab = ["feat_7", "feat_17", "feat_27",
             "feat_72", "feat_84",
             "feat_96", "feat_108"]

    g8 = sns.heatmap(cmap="inferno", data=similarity, linewidths=.5, vmax=1)
    g8.set_xticklabels(x_lab)
    g8.set_yticks([x + .5 for x in range(0, 14, 2)])
    g8.set_yticklabels(y_lab)

    plt.show()


def hist_main_features(data):
    """
    This function is used to plot histograms about main features distributions,
    particularly the categorical features: action and feature 0.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # matrix containing only the most relevant features
    data_main = data.loc[:, ["resp", "resp_1", "resp_2",
                             "resp_3", "resp_4", "weight", "weighted_resp", "action", "date"]]
    fig, axes = plt.subplots(2, 4)

    fig.suptitle('Histogram main features', fontsize=20)
    i = 0
    j = 0
    data_main.drop(["date"], axis=1, inplace=True)
    for column in data_main.columns:
        if column == "action":
            sns.histplot(ax=axes[i, j],
                         data=data_main,
                         x=column,
                         stat="probability",
                         bins=3)
        else:
            sns.histplot(ax=axes[i, j],
                         data=data_main,
                         x=column,
                         stat="probability",
                         binrange=(-data_main[column].std(), data_main[column].std()))
        i = i+1
        if i == 2:
            i = 0
            j = j+1

    save_oneplot_options("Figures/histogram_main.png")
    plt.show()


def hist_anon_features(data):
    """
    This function is used to plot histograms about anonymous features distributions.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # built matrix containing the anonymous features
    data_anon = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
                           "weight", "weighted_resp", "action", "ts_id"], axis=1)
    names = []  # empty list, it will store the names of the png files
    plt.subplots(3, 3, figsize=(6, 6))
    print("Working on plot the histograms of the anonymous features...\n")
    # create 14 images 3x3 containig plot of each anonymous features
    for i in range(14):
        data_hist = data_anon.iloc[:, (9*i):(9*i+9)]
        # to better visualize the results we set the range of each 3x3 histogram
        # equal to the max variance between the features
        max_var = data_hist.var().max()
        data_hist.plot(subplots=True, layout=(
            3, 3), figsize=(6., 6.), kind="hist", bins=100, yticks=[],
            range=([-max_var, max_var]))
        plt.subplots_adjust(wspace=0.4)
        plt.suptitle("Distributions for anonymous features", fontsize=20)
        # storing names of png files in case we want to save them
        names.append('Figures/histogram_anonimous_features{}.png'.format(i))
    save_plots_options(names)


def boxplot_main(data):
    """
    This function is used to build the boxplot of main features. The user
    can decide if save or not the figure obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # matrix containing only the most relevant features
    data_main = data.loc[:, ["resp", "resp_1", "resp_2",
                             "resp_3", "resp_4", "weighted_resp", ]]
    print("Working on boxplot of features...\n")

    # Plot th boxplot (outliers aren't shown)
    sns.catplot(kind="boxen",
                data=data_main,
                orient="h",
                outlier_prop=0.05)

    plt.title("Boxen plot main features", fontsize=18)
    plt.legend()
    save_oneplot_options("Figures/boxplot_main.png")


def save_data_options(data, name_save):
    """
    This fuction is used for the save options of a dataframe as .cvs.
    Parameters
    ----------
    object: Pandas dataframe
        Dataframe we want to save or not.
    name_save: string
        Figure's name.
    """
    save_flag = False
    while save_flag is False:
        save = input("Done. \nDo you want to save it?\ny/n\n")
        if save == "y":
            # save new matrix as csv
            data.round(3).to_csv(name_save)
            print("Saved successfully as {}\n".format(name_save))
            save_flag = True
        elif save == "n":
            save_flag = True
        else:
            print("Please enter valid key.\n")


def save_oneplot_options(name_save):
    """
    This fuction is used for the save options and visualization of a single plot.
    Parameters
    ----------
    name_save: string
        Figure's name.
    """
    save_flag = False
    while save_flag is False:
        # reads from input if we need to save the plot
        save = input("Done. \nDo you want to save it?\ny/n\n")
        if save == "y":
            plt.savefig(name_save, dpi=300)
            print("Saved successfully as {}\n".format(name_save))
            save_flag = True
        elif save == "n":
            save_flag = True
        else:
            print("Please enter valid key.\n")
        plt.show()
        plt.close("all")


def save_plots_options(names_save):
    """
    This fuction is used for the save options and visualization of some plots.
    Parameters
    ----------
    names_save: list
        List in which we stored figures' names.
    """
    save_flag = False
    while save_flag is False:
        save = input("Done. \nDo you want to save them?\ny/n\n")
        if save == "y":
            for name in names_save:
                plt.savefig(name, dpi=300)
            print("Saved successfully.\n")
            save_flag = True
        elif save == "n":
            save_flag = True
        else:
            print("Please enter valid key.\n")
        plt.show()
        plt.close("all")



if __name__ == '__main__':
    # start time for the exection of this main
    start = time.time()
    # import the choosen dataset
    comp_data = initial_import.main()
    # compute the maximum value of u possible
    u_val, tradng_days = compute_profit(comp_data)
    # user window
    FLAG = False  # used to make sure to go back once an invalid string is entered
    while FLAG is False:
        value = activity_choice()
        if value == "1":
            # compute matrix containig useful statistical informations
            stats = statistical_matrix(comp_data)
            save_data_options(stats, "Matrices/stats_complete.csv")
        elif value == "2":
            plot_main_features(comp_data)
        elif value == "3":
            correlation_analysis(comp_data)
        elif value == "4":
            plot_anon_features(comp_data)
        elif value == "5":
            missing_data_analysis(comp_data, .005)
        elif value == "6":
            hist_main_features(comp_data)
        elif value == "7":
            hist_anon_features(comp_data)
        elif value == "8":
            boxplot_main(comp_data)
        elif value == "9":
            print("End session.\n")
            FLAG = True
        else:
            print("Please enter valid key\n")

    # compute execution time
    mins = (time.time()-start)//60
    sec = int((time.time()-start) % 60)
    print('Execution time is: {} min {} sec\n'.format(mins, sec))
