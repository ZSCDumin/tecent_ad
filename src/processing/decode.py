import pandas as pd
import numpy as np
from Configs import _C as cfg

import torch, os

"""
Train creative_id number is 2481135
Train user_id number is 900000
Train ad_id number is 2264190
Train product_id number is 33272
Train product_category number is 18
Train advertiser_id number is 52090
Train industry number is 325
"""
    

def ad_decode(cols=None):
    
    ad_train = pd.read_csv(cfg.train_dir + "ad.csv")
    ad_test  = pd.read_csv(cfg.test_dir  + "ad.csv")
    ad_train.fillna('nan', inplace=True)
    ad_train.replace('\\N', 'nan', inplace=True)
    ad_test.fillna('nan', inplace=True)
    ad_test.replace('\\N', 'nan', inplace=True)

    cols = cols if cols else ad_train.columns.tolist()
    for col in cols:
        save_dir = cfg.decode + f"all_{col}_decode_string.pth"
        if os.path.exists(save_dir):
            print("已经存在 `{}`\n".format(save_dir))
            continue

        id_set = set(ad_train.loc[ad_train[col] != 'nan', col].astype('int64').astype('str').tolist())
        id_list = list(id_set)

        _, decode = pd.factorize(id_list, sort=True)
        print("Train {} number is {}".format(col, len(decode)))

        id_list_extra = [id_str for id_str in ad_test.loc[ad_test[col] != 'nan', col].astype('int64').astype('str').unique() if id_str not in id_set]
        _, decode_extra = pd.factorize(id_list_extra, sort=True)

        decode_string = ','.join(['pad', 'nan'] + decode.tolist() + decode_extra.tolist())
        torch.save(decode_string, save_dir)
        print("all_{}_decode_string: {} \n"
              "Saving at `{}`\n".format(col, decode_string[:60], save_dir))


def click_decode(cols=None):

    cols = cols if cols else ['creative_id', 'user_id']

    click_train = pd.read_csv(cfg.train_dir + "click_log.csv")
    click_test  = pd.read_csv(cfg.test_dir + "click_log.csv")

    for col in cols:

        save_dir = cfg.decode + f"all_{col}_decode_string.pth"
        if os.path.exists(save_dir):
            print("已经存在 `{}`".format(save_dir))
            continue

        id_set = set(click_train[col].astype('int64').astype('str'))
        id_list = list(id_set)

        _, decode = pd.factorize(id_list, sort=True)
        print("Train {} number is {}".format(col, len(decode)))

        id_list_extra = [id_str for id_str in click_test[col].astype('int64').astype('str').unique() if id_str not in id_set]
        _, decode_extra = pd.factorize(id_list_extra, sort=True)

        decode_string = ','.join(['pad', 'nan'] + decode.tolist() + decode_extra.tolist())
        torch.save(decode_string, save_dir)
        print("all_{}_decode_string: {} \n"
              "Saving at `{}`\n".format(col, decode_string[:60], save_dir))


if __name__ == "__main__":
    click_decode()
    ad_decode()
