import pandas as pd
def findAllZero(df: pd.DataFrame,factor:str,time:str) -> pd.DataFrame:
    segments = []
    start = None
    for index,_ in df.iterrows():
        if df.loc[index,factor] == 0:
            if start is None:
                start = df.loc[index,time]
        else:
            if start is not None:
                segments.append([start,df.loc[index,time]])
                start = None
    if start is not None:
        segments.append([start,df.loc[len(df)-1,time]])
    stop = pd.DataFrame(columns=['start','end'],data = segments)
    return stop

def deleteZeroDf(df:pd.DataFrame,delete:pd.DataFrame)->pd.DataFrame:
    for index,_ in delete.iterrows():
        start = delete.loc[index,'start']
        end = delete.loc[index,'end']
        df = df[(df['timestamps'] < start) | (df['timestamps'] > end)]
    return df

def speedLabel(speed):
    if (speed <60): return "low"
    if (speed >= 60) & (speed < 80): return "mid"
    else: return "high"