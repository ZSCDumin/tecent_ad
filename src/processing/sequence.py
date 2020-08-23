import os, pdb, torch
import pandas as pd
import numpy as np

from Configs import _C as cfg
from .click import make_click_with_ad


def make_user_behavior_seq_test(cols = ['creative_id', 'ad_id', 'advertiser_id',]):

    save_dir = cfg.features + "user_behavior_df_test.pkl"
    if os.path.exists(save_dir):
        print("已存在 `{}`, 直接读取返回...".format(save_dir))
        user_behavior_df_test=pd.read_pickle(save_dir)
        return user_behavior_df_test

    print("\n无本地 user_behavior_df_test, 重新生成 `{}`".format(save_dir))
    click_all = make_click_with_ad()
    click_all.replace('\\N', 'nan', inplace=True)

    for col in cols:
        decode_list = torch.load(cfg.decode + f"all_{col}_decode_string.pth").split(',')
        decode_dict = {v: str(i) for i, v in enumerate(decode_list)}

        click_all[col] = click_all[col].astype('str').map(decode_dict)
        print("all_{}_decode_list: {}".format(col, decode_list[:20]))
        print("click_all[`{}`]: {}".format(col, click_all[col][:60]))

    click_test  = click_all.query("type=='test'").sort_values(["time"]).reset_index(drop=True)

    click_test['user_id'] = click_test['user_id'].astype('int64')

    user_group = click_test.groupby(['user_id'])
    user_behavior_df_test = pd.DataFrame()
    for col in cols:
        df = user_group.agg({col: list})
        user_behavior_df_test[col + "_200"] = df[col].map(lambda sentence: ','.join(sentence))

    user_behavior_df_test['seq_length'] = user_group[cols[0]].agg('count').values

    user_behavior_df_test.reset_index().to_pickle(save_dir)

    return user_behavior_df_test


def make_user_behavior_seq(cols = ['creative_id', 'ad_id', 'advertiser_id',]):

    save_dir = cfg.features + "user_behavior_df.pkl"
    if os.path.exists(save_dir):
        print("已存在 `{}`, 直接读取返回...".format(save_dir))
        user_behavior_df=pd.read_pickle(save_dir)
        return user_behavior_df

    print("\n无本地 user_behavior_df, 重新生成 `{}`".format(save_dir))
    click_all = make_click_with_ad()
    click_all.replace('\\N', 'nan', inplace=True)

    for col in cols:
        decode_list = torch.load(cfg.decode + f"all_{col}_decode_string.pth").split(',')
        decode_dict = {v: str(i) for i, v in enumerate(decode_list)}

        click_all[col] = click_all[col].astype('str').map(decode_dict)
        print("all_{}_decode_list: {}".format(col, decode_list[:20]))
        print("click_all[{}]: {}".format(col, click_all[col][:60]))
        if click_all[col].isna().any():
            pdb.set_trace()

    click_train = click_all.query("type=='train'").sort_values(["time"]).reset_index(drop=True)
    click_test  = click_all.query("type=='test'").sort_values(["time"]).reset_index(drop=True)

    click_train['user_id'] = click_train['user_id'].astype('int64')

    user_group = click_train.groupby(['user_id'])
    user_behavior_df = pd.DataFrame()
    for col in cols:
        df = user_group.agg({col: list})
        user_behavior_df[col + "_200"] = df[col].map(lambda sentence: ','.join(sentence))

    user_behavior_df['seq_length'] = user_group[cols[1]].agg('count').values

    user = pd.read_csv(cfg.train_dir+'/user.csv')
    user['user_id'] = user['user_id'].astype('int64')
    user_behavior_df = user_behavior_df.reset_index().merge(user, how="left", on="user_id")
    user_behavior_df.to_pickle(save_dir)

    return user_behavior_df


def clip_fn(sentence):
    length = len(sentence)
    if length >= 200:
        sentence = sentence[-200:]
    else:
        sentence += [0] * (200 - length)
    return ','.join(sentence)


def seq2lengths(seq):
    length = (np.array(seq) != 0).sum()
    return length


def seq2string_with_length(seq):
    length = len(seq)
    seq_string = ",".join(str(w) for w in seq)
    return f"{seq_string}_{length}"



""" 
creative_id, 3412772 | 931637
ad_id, 3027360 | 763170
user_id, 1900000
click_times, 94 | 53
product_id, 39057 | 5784
product_category, 18 | None
advertiser_id, 57870
industry, 332 | {8, 89, 93, 151, 195, 196}
category 1-18, 无需编码
"""
