import pandas as pd

if __name__ == '__main__':
    dfs = []
    for i in range(1,9):
        dfs.append(pd.read_csv("search{i}kbest.csv".format(i=i)))

    for i in range(1,8):
            dfs.append(pd.read_csv("search{i}pca.csv".format(i=i)))

    big_frame=pd.concat(dfs,ignore_index=True)
    big_frame = big_frame.sort_values(by="mean_test_score", ascending=False)
    big_frame.to_csv("all_searches.csv")
