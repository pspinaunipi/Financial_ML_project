"""
In this module we implemented some functions for the exploratory data analysis.
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
colorMap = sns.light_palette("blue", as_cmap=True)
import dabl
import datatable as dt
import gc
from initial_import import import_training_set
from scipy.optimize import curve_fit
import plotly.express as px


def cum_resp(data):
    """
    This function computes and plots the different cumulative resps
    obtained by considering four different time horizons.
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    #compute the different cumulative resps
    resp    = pd.Series(1+(data.groupby('date')['resp'].mean())).cumprod()
    resp_1  = pd.Series(1+(data.groupby('date')['resp_1'].mean())).cumprod()
    resp_2  = pd.Series(1+(data.groupby('date')['resp_2'].mean())).cumprod()
    resp_3  = pd.Series(1+(data.groupby('date')['resp_3'].mean())).cumprod()
    resp_4  = pd.Series(1+(data.groupby('date')['resp_4'].mean())).cumprod()
    #plot the trends
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_title ("Cumulative resp with different time horizons")
    resp.plot(label='resp')
    resp_1.plot(label='resp_1')
    resp_2.plot(label='resp_2')
    resp_3.plot(label='resp_3')
    resp_4.plot(label='resp_4')
    # day 85 marker
    ax.axvline(x=85, linestyle='--', alpha=0.3, c='red', lw=1)
    plt.legend(loc="lower left")
    plt.show()
    gc.collect()

def cum_return(data):
    """
    This function computes and visualizes the cumulative daily return over time,
    which is given by weight multiplied by the value of the relative resp.
    """
    #compute return for the four different time horizons
    data['weight_resp']   = data['weight']*data['resp']
    data['weight_resp_1'] = data['weight']*data['resp_1']
    data['weight_resp_2'] = data['weight']*data['resp_2']
    data['weight_resp_3'] = data['weight']*data['resp_3']
    data['weight_resp_4'] = data['weight']*data['resp_4']

    #plot the returns obtained
    fig, ax = plt.subplots(figsize=(15, 5))
    resp    = pd.Series(1+(data.groupby('date')['weight_resp'].mean())).cumprod()
    resp_1  = pd.Series(1+(data.groupby('date')['weight_resp_1'].mean())).cumprod()
    resp_2  = pd.Series(1+(data.groupby('date')['weight_resp_2'].mean())).cumprod()
    resp_3  = pd.Series(1+(data.groupby('date')['weight_resp_3'].mean())).cumprod()
    resp_4  = pd.Series(1+(data.groupby('date')['weight_resp_4'].mean())).cumprod()
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_title ("Cumulative daily return")
    resp.plot(label='resp x weight')
    resp_1.plot(label='resp_1 x weight')
    resp_2.plot(label='resp_2 x weight')
    resp_3.plot(label='resp_3 x weight')
    resp_4.plot(label='resp_4 x weight')
    # day 85 marker
    ax.axvline(x=85, linestyle='--', alpha=0.3, c='red', lw=1)
    plt.legend(loc="lower left")
    plt.show()



def resp_action_dist(data):
    """
    This function is used to visualize the histogram for the resp distribution and
    the histogram for the weighted resp distribution. We also compute min and max
    values, skew and kurtosis.
    In addition we also visualize the distribution of the action between the two
    values 0 and 1, because we need to know if our classes are balanced.
    """

    plt.figure(figsize = (12,5)
    #histogram for resp distribution
    ax = sns.distplot(data['resp'],
             bins=3000,
             kde_kws={"clip":(-0.05,0.05)},
             hist_kws={"range":(-0.05,0.05)},
             color='darkcyan',
             kde=False);
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    plt.xlabel("Resp values distribution", size=14)
    plt.show();

    #compute information about the resp distribution
    min_resp = data['resp'].min()
    print('The minimum value for resp is: %.5f' % min_resp)
    max_resp = data['resp'].max()
    print('The maximum value for resp is:  %.5f' % max_resp)
    print("Skew of resp is:      %.2f" %data['resp'].skew() )
    print("Kurtosis of resp is: %.2f"  %data['resp'].kurtosis() )


    #discard zero weights
    data_no0 = data.query('weight > 0').reset_index(drop = True)
    data_no0['weight_Resp'] = data_no0['weight'] * (data_no0['resp'])
    #plot histogram for weighted resp distribution
    plt.figure(figsize = (12,5))
    ax = sns.distplot(data_no0['weight_Resp'],
             bins=1500,
             kde_kws={"clip":(-0.02,0.02)},
             hist_kws={"range":(-0.02,0.02)},
             color='darkcyan',
             kde=False);
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    plt.xlabel("Weighted resp distribution", size=14)
    plt.show()

    #compute information about the weighted resp distribution
    min_wresp = data_no0['weight_Resp'].min()
    print('The minimum value for weighted resp is: %.5f' % min_wresp)
    max_wresp = data_no0['weight_Resp'].max()
    print('The maximum value for weighted resp is:  %.5f' % max_wresp)
    print("Skew of weighted resp is:      %.2f" %data_no0['weight_Resp'].skew() )
    print("Kurtosis of weighted resp is: %.2f"  %data_no0['weight_Resp'].kurtosis())

    #plot histogram for action
    sns.histplot(data=data,
                 x="action",
                 stat="probability",
                 bins=3)
    plt.show()


def weight_dist(data):
    """
    In this function we explore the weigts distribution and also compute the
    percentage of zero weights.
    """
    #compute percentage of zero weights
    zeros = (100/data.shape[0])*((data.weight.values == 0).sum())
    print('Percentage of zero weights is: %i' % zeros +"%")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    #plot histogram only for non-zero weights
    sns.distplot(data.query('weight != 0')["weight"], ax=axs[0])
    axs[0].set_title("Weights Distribution")
    #boxplot only for non-zero weights
    sns.boxplot(x="weight", data=data[data["weight"]!=0], ax=axs[1])
    axs[1].set_title("Weights Box plot")

    plt.show()

    #compute information about the weights distribution
    min_weight = data['weight'].min()
    print('The minimum value for weight is: %.5f' % min_weight)
    max_weight = data['weight'].max()
    print('The maximum value for weight is:  %.5f' % max_weight)
    print("Skew of weight is:      %.2f" %data['weight'].skew() )
    print("Kurtosis of weight is: %.2f"  %data['weight'].kurtosis() )

def trades_days(data):
    """
    This function in used to show the total number of trades per day and the
    relative histogram of the number of trades per day.
    """

    #plot trades per day
    plt.figure(figsize = (12,5))
    trades_per_day = data.groupby(['date'])['ts_id'].count()
    plt.plot(trades_per_day)
    plt.xlabel ("Day", fontsize=18)
    plt.ylabel ("Total number of ts_id for each day", fontsize=18)
    plt.show()

    #plot histogram for number of trades per day
    plt.figure(figsize = (12,5))
    ax=sns.distplot(trades_per_day,
             bins=125,
             kde_kws={"clip":(1000,20000)},
             hist_kws={"range":(1000,20000)},
             color='darkcyan',
             kde=True);
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    plt.xlabel("Number of trades per day", size=14)
    plt.show()


def resp_analysis(data):
    """
    This function compute the trends for our different resp and then how change
    the standard deviations for the resps at different time horizons.
    """
    plt.figure(figsize=(12,5))
    resp_mean = data.groupby('date')[['resp_1', 'resp_2', 'resp_3', 'resp_4']].mean().plot(color=["black","red","green","blue"])
    sns.lineplot(x=[0,500],y=[0,0],lw=1.5)
    plt.title("Mean values of resps over time")
    plt.show()

    plt.figure(figsize=(12,5))
    resp_std = data.groupby('date')[['resp_1', 'resp_2', 'resp_3', 'resp_4']].std().plot(color=["gray","blue","green","red"])
    plt.title("Resp stardard deviation for each trading day ")
    plt.show()


def utility_score(data):
    """
    Thi function in used to compare the utility scores over five different action
    strategies:
    1)Taking no actions
    2)Taking all actions
    3)Taking all actions with a return value greater than the daily volatility
    4)Taking all actions with a return value greater than the daily signal-to-noise ratio
    5)Taking all actions with a return value greater than the overall signal-to-noise ratio
    """
    i_ranges = 499
    plt.figure(figsize=(12,5))
    ################

    u_collected = np.zeros(i_ranges)

    for i_range in (range(i_ranges)):

        dates = np.linspace(0, i_range, i_range+1).astype(int)
        threshold = 0
        p = np.zeros(len(dates))
        for i in dates:
            data_i = data[data.date == i]
            weight_i = data_i.weight
            resp_i = data_i.resp
            action_i = data_i.resp.apply(lambda x: 1 if x > threshold else 0)
            p[i] = np.sum(weight_i * resp_i * action_i)

        t = (np.sum(p) / np.sqrt(np.sum(p**2))) * np.sqrt(250/len(dates))
        u = np.minimum(np.maximum(t, 0), 6) * np.sum(p)

        u_collected[i_range] = u

    plt.plot(range(i_ranges),u_collected, 'b', label='Only Positive Response')

    ######################

    u_collected = np.zeros(i_ranges)

    for i_range in (range(i_ranges)):

        dates = np.linspace(0, i_range, i_range+1).astype(int)
        threshold = data[data.date.isin(dates)].resp.mean()/data[data.date.isin(dates)].resp.std()
        p = np.zeros(len(dates))
        for i in dates:
            data_i = data[data.date == i]
            weight_i = data_i.weight
            resp_i = data_i.resp
            action_i = data_i.resp.apply(lambda x: 1 if x > threshold else 0)
            p[i] = np.sum(weight_i * resp_i * action_i)

        t = (np.sum(p) / np.sqrt(np.sum(p**2))) * np.sqrt(250/len(dates))
        u = np.minimum(np.maximum(t, 0), 6) * np.sum(p)

        u_collected[i_range] = u

    plt.plot(range(i_ranges),u_collected, 'r', label='Greater Response than Signal-to-Noise')
######################

    u_collected = np.zeros(i_ranges)
    threshold = data.resp.mean()/data.resp.std()

    for i_range in (range(i_ranges)):

        dates = np.linspace(0, i_range, i_range+1).astype(int)
        p = np.zeros(len(dates))
        for i in dates:
            data_i = data[data.date == i]
            weight_i = data_i.weight
            resp_i = data_i.resp
            action_i = data_i.resp.apply(lambda x: 1 if x > threshold else 0)
            p[i] = np.sum(weight_i * resp_i * action_i)

        t = (np.sum(p) / np.sqrt(np.sum(p**2))) * np.sqrt(250/len(dates))
        u = np.minimum(np.maximum(t, 0), 6) * np.sum(p)
        u_collected[i_range] = u

    plt.plot(range(i_ranges),u_collected, 'y', label='Greater Response than Overall Signal-to-Noise')
######################

    u_collected = np.zeros(i_ranges)

    for i_range in (range(i_ranges)):
        dates = np.linspace(0, i_range, i_range+1).astype(int)
        p = np.zeros(len(dates))
        for i in dates:
            data_i = data[data.date == i]
            weight_i = data_i.weight
            resp_i = data_i.resp
            action_i = 1
            p[i] = np.sum(weight_i * resp_i * action_i)

        t = (np.sum(p) / np.sqrt(np.sum(p**2))) * np.sqrt(250/len(dates))
        u = np.minimum(np.maximum(t, 0), 6) * np.sum(p)
        u_collected[i_range] = u

    plt.plot(range(i_ranges),u_collected, 'g', label='All Actions')


######################

    u_collected = np.zeros(i_ranges)
    plt.plot(range(i_ranges),u_collected, 'm', label='No actions')


######################
    plt.legend()
    plt.title('Utility Score u', fontsize=16)
    plt.xlabel('|i|', fontsize=16)
    plt.show()

    data['p_i'] = data['weight'] * data['resp']
    data['p_i^2'] = (data['p_i']**2)
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    sns.scatterplot(data=data, x='p_i^2', y='p_i', ax=ax)
    ax.set_title('Return over Risk', fontsize=16)
    ax.set_xlabel('Volatility ($p_i^2$)', fontsize=16)
    ax.set_ylabel('Return ($p_i$)', fontsize=16)

    plt.show()


if __name__ == '__main__':
    # load dataset
    data = import_training_set(fast_pc = True)
    data.dropna(inplace=True)
    cum_resp(data)
    cum_return(data)
    resp_action_dist(data)
    weight_dist(data)
    trades_days(data)
    resp_analysis(data)
    utility_score(data)
