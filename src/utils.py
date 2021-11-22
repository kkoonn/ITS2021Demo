import pandas as pd

# 指定された日付の範囲のインデックスを返す
def getIndexByRangeDate(df, dt1, dt2):
    mask1 = dt1 <= df['SBSDepartureTime']
    mask2 = df['SBSDepartureTime'] < dt2
    index1 = df[mask1 & mask2].index[0]
    index2 = df[mask1 & mask2].index[-1]
    return index1, index2