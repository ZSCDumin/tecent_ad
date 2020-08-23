import os, pdb, gc
import numpy as np
import pandas as pd
from Configs import _C as cfg


def make_click_with_ad():
    """ 拼接点击日志和广告属性,
        并存储本地, 暂时过滤了测试集的非登入 creative_id 的日志记录
    """

    save_dir = cfg.features + "click_all.pkl"
    if os.path.exists(save_dir):
        print("已经存在 click_all.pkl, 直接读取`{}`\n".format(save_dir))
        click_all = pd.read_pickle(save_dir)
        return click_all

    click_train = pd.read_csv(cfg.train_dir + "click_log.csv")
    click_test  = pd.read_csv(cfg.test_dir  + "click_log.csv")
    
    click_train['creative_id']=click_train['creative_id'].astype('int64')
    click_test['creative_id']=click_test['creative_id'].astype('int64')

    print("过滤测试集未登入训练集 creative_id 记录\n")
    click_test  = click_test.loc[click_test['creative_id'].isin(click_train['creative_id'].unique())]

    ad_train = pd.read_csv(cfg.train_dir + "ad.csv")
    ad_train['creative_id'] = ad_train['creative_id'].astype('int64')
    click_log = click_train.merge(ad_train, how="left", on="creative_id")
    # click_log = click_log.sort_values(["time"]).reset_index(drop=True)
    click_log["type"] = "train"

    ad_test = pd.read_csv(cfg.test_dir  + "ad.csv")
    ad_test['creative_id'] = ad_test['creative_id'].astype('int64')
    click_log_test = click_test.merge(ad_test, how="left", on="creative_id")
    # click_log_test = click_log_test.sort_values(["time"]).reset_index(drop=True)
    click_log_test['type'] = "test"

    click_all = click_log.append(click_log_test)
    click_all.to_pickle(save_dir)

    return click_all
