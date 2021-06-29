import pandas as pd

if __name__ == '__main__':
    dfs = []
    for i in range(1,45):
        dfs.append(pd.read_csv("search_random{i}.csv".format(i=i)))


    big_frame=pd.concat(dfs,ignore_index=True)
    big_frame = big_frame.sort_values(by="mean_test_f1", ascending=False)
    big_frame.to_csv("all_searches.csv")
