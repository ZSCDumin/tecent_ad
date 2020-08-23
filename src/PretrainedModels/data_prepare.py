'''
第一步:数据准备
下载的数据整合成训练和测试需要的数据
生成的时序数据文件为
seq-all 
seq-train
seq-test
格式user_id,x,y
x为输入的时序，时序使用广告商id和产品id进行拼接，重合度99%
y为预测标签值，
'''
import os, gc,torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Configs import _C as cfg


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


print("训练集读取")
click_train = pd.read_csv(cfg.train_dir + "click_log.csv")
ad_train = pd.read_csv(cfg.train_dir + "ad.csv")
click_log = click_train.merge(ad_train, how="left", on="creative_id", )
click_log["type"] = "train"

del(click_train, ad_train)
gc.collect()


print("测试读取")
click_test = pd.read_csv(cfg.test_dir + "click_log.csv")
ad_test = pd.read_csv(cfg.test_dir + "ad.csv")
click_log_test = click_test.merge(ad_test, how="left", on="creative_id", )
click_log_test['type'] = "test"

del(click_test, ad_test)
gc.collect()

click_all = click_log.append(click_log_test)
del(click_log, click_log_test)
gc.collect()

print("原始的数据保存"))
click_all.to_pickle(cfg.features + "click-raw.pkl")


print("0填充")
click_all.replace('\\N', 0)
click_all.fillna(0, inplace=True)
# click_all["product_id"]=click_all['product_id'].apply(lambda x : 0 if x == '\\N' else x)
# click_all["industry"]=click_all['industry'].apply(lambda x : 0 if x == '\\N' else x)
click_all["product_id"]=click_all["product_id"].astype(np.int32)
click_all["industry"]=click_all["industry"].astype(np.int32)
click_all["type"]=click_all["type"].astype(str)
click_all = reduce_mem_usage(click_all)
click_all.to_pickle(cfg.features + "click-fill0.pkl")


print("载入用户标签")
user_label = pd.read_csv(cfg.train_dir + "user.csv")
print("年龄和性别合并")
user_label["demand"]=(user_label["gender"]-1)*10+(user_label["age"]-1)
user_label["demand"]=user_label["demand"].astype(np.int32)
user_label = reduce_mem_usage(user_label)
print("合并")
data_all=click_all.merge(user_label, how="left", on="user_id", )

print("测试的预测数据填充0")
data_all["age"]=data_all["age"].fillna(0)
data_all["gender"]=data_all["gender"].fillna(0)
data_all["age"]=data_all["age"].astype(np.int32)
data_all["gender"]=data_all["gender"].astype(np.int32)
data_all = reduce_mem_usage(data_all)
data_all.to_pickle(cfg.features + "all.pkl")


print("advertiser_id和product_id")
data_all["adv_pro"]=data_all["advertiser_id"].astype(str)+":"+data_all["product_id"].astype(str)

data_all.to_pickle("data/df-fixed.pkl")

'''
creative_id	ad_id	product_id	product_category	advertiser_id	industry
adv_pro:advertiser_id - product_id
'''
cate_cols=["adv_pro","ad_id","product_id","advertiser_id","creative_id","product_category","industry"]
for col in cate_cols:
    print(col,data_all[col].nunique())
    print("编码")
    le = LabelEncoder()
    data_all[col]=le.fit_transform(data_all[col].astype("str"))
    data_all[col]=data_all[col]+1 #加1是因为0是padding
    data_all[col]=data_all[col].astype("str")
    torch.save(le, cfg.decode + f"{col}.pkl")

print(data_all.info())
#全部编码后的文件
data_all.to_pickle(cfg.features + "all-encoded.pkl")

print("生成训练和验证时序文件")
#按照日期排序
data_all=data_all.sort_values(by="time")


#生成时序列
seq_df={}
for col in cate_cols:
    print("训练和验证时序文件",col)
    seq_df[col]=data_all[['user_id',col]].groupby(['user_id']).apply(lambda x:",".join(x[col])).reset_index()
    seq_df[col]=seq_df[col].rename(columns={0: col})

print("序列合并标签")


seq_all=None
for key in seq_df:
    if seq_all is None:
        print("初始化",key)
        seq_all=seq_df[key].copy()
    else:
        print("合并",key)
        seq_all=seq_all.merge(seq_df[key], how="left", on="user_id", )
print("合并用户信息")
seq_all=seq_all.merge(user_label, how="left", on="user_id", )
print("生成序列长度列")
seq_all["seq_len"]=seq_all["adv_pro"].apply(lambda x: len(x.split(",")))

'''
print("对年龄进行编码，变成二元分类问题")
for i in range(1,11):
    seq_all["a"+str(i)]=seq_all["age"].apply(lambda x : 1 if x==i else 0)
'''

print(seq_all.info())
#print(seq_all.head())

print("保存时序数据")
seq_all.to_pickle("data/seq-all.pkl")
seq_all.sample(n=10).to_csv(cfg.features + "seq-sample.csv")

print("总数据",len(seq_all))
seq_test=seq_all[seq_all["user_id"]>900000]
seq_test.to_pickle(cfg.features + "seq-test.pkl")
print("test",len(seq_test))

seq_train=seq_all[seq_all["user_id"]<=900000]
seq_train.to_pickle(cfg.features + "seq-train.pkl")
print("test",len(seq_train))

exit()

seq_val=seq_train.sample(n=10000,random_state=999999)
seq_train=seq_train.drop(seq_val.index)
seq_train.to_csv(cfg.features + "seq-train.csv", index=False)
seq_val.to_csv(cfg.features + "seq-val.csv",index=False)
print("seq_train",len(seq_train))
print("seq_val",len(seq_val))
