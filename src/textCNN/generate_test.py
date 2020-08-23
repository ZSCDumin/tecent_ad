import torch, os, gc
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
import time
from multiprocessing import Pool

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df

def getFre(df):
    fre = df['creative_id'].value_counts()
    for i in fre.index:
        t1 = df[df['creative_id']==i]
        S = t1['click_times'].sum()
        fre[i] = S
    #print('\r%d'%df['user_id'][0],end='')
    return fre.sort_values(ascending=False)

if __name__ == '__main__':
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

    test_dir = r"F:\tx2020\data\test"
    click_train = pd.read_csv(test_dir + "\\click_log.csv")
    click_train = reduce_mem_usage(click_train)
    ad_train = pd.read_csv(test_dir + "\\ad.csv")
    ad_train = reduce_mem_usage(ad_train)

    #填充
    for col in ["product_id", "industry"]:
        ad_train[col] = ad_train[col].replace("\\N", -1).map(int)

    #合并点击及广告信息
    train_data= click_train.merge(ad_train, how="left", on="creative_id", )
    del(click_train,ad_train)
    gc.collect()

    train_data.sort_values(['user_id','time','click_times'],inplace=True)
    div = train_data['user_id'].value_counts()
    lst = []
    i = 0
    starttime = time.time()
    for t in range(3000001,4000001):
        s = div[t]
        lst.append(train_data.iloc[i:i+s])
        i += s
        if t%10 == 0:
            print('\r %.5f'%((t-3000000)/1000000),end="")
    endtime = time.time()
    dtime = endtime - starttime
    print(dtime)

    starttime = time.time()
    p = Pool(8)
    lst2 = p.map(getFre, lst)
    p.close()
    p.join()
    endtime = time.time()
    dtime = endtime - starttime
    print(dtime)
    joblib.dump(lst2,'Testdata.pkl')

