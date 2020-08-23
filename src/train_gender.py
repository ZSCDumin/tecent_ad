import random, multiprocessing, os, gc, pdb
import pandas as pd, numpy as np
from datetime import datetime
from tqdm import tqdm
from Configs import _C as cfg

from processing.decode import click_decode, ad_decode
from processing.click import make_click_with_ad
from processing import make_user_behavior_seq, make_user_behavior_seq_test

import torch
from torch import nn
# from gensim.models import Word2Vec

from deepmodels import User_statisic_att_model
from deepmodels.inputs import VarLenSparseFeat, SparseFeat


def get_feature(df, fc):
    array = [clip_seq(seq, length, fc.maxlen)
             for seq, length in zip(df[fc.name + "_200"], df["seq_length"])]
    return np.asarray(array).astype(fc.dtype)

def clip_seq(seq, length, maxlen):
    sequence = seq.split(',') # if len(sequence) != length: pdb.set_trace()
    if length > maxlen:
        return sequence[-maxlen:]
    else:
        return sequence + ['0'] * (maxlen - length) # seq.split(',')[-fc.maxlen:] + ['0'] * (fc.maxlen - length) 
    

if __name__ == "__main__":

    embedding_dim = 64
    feature_columns=[
        VarLenSparseFeat(
            SparseFeat('creative_id', 3412772 + 2, embedding_dim = embedding_dim),
            maxlen=200, length_name='seq_length'),
        VarLenSparseFeat(
            SparseFeat('advertiser_id', 57870 + 2, embedding_dim = embedding_dim),
            maxlen=200, length_name='seq_length'),
        VarLenSparseFeat(
            SparseFeat('ad_id', 3027360 + 2, embedding_dim = embedding_dim),
            maxlen=200, length_name='seq_length'),
        # VarLenSparseFeat(SparseFeat('industry', 331 + 2, embedding_dim = embedding_dim), maxlen=200, length_name='seq_length'),
    ]

    if os.path.exists(cfg.features + "y_gender.pth"):
        X = torch.load(cfg.features + "X.pth")
        y = torch.load(cfg.features + "y_gender.pth")
        X_test = torch.load(cfg.features + "X_test.pth")
    else:
        if False: # 手动构建编码文件 如果本地不存在
            click_decode()
            ad_decode()

        user_behavior_df = make_user_behavior_seq()
        X = {fc.name: get_feature(user_behavior_df, fc)
             for fc in feature_columns}
        X["seq_length"] = torch.from_numpy(user_behavior_df["seq_length"].values)
        y = torch.from_numpy(user_behavior_df['gender'].values)

        user_behavior_df_test = make_user_behavior_seq_test()
        X_test = {fc.name: get_feature(user_behavior_df_test, fc)
                  for fc in feature_columns}
        X_test["seq_length"] = torch.from_numpy(user_behavior_df_test["seq_length"].values)
        
        torch.save(X, cfg.features + "X.pth")
        torch.save(y, cfg.features + "y_gender.pth")
        torch.save(X_test, cfg.features + "X_test.pth")

    pdb.set_trace()
    y = y-1 # 标签转换 (1, 2) -> (0, 1)

    model = User_statisic_att_model(feature_columns=feature_columns)
    model.compile(optimizer=torch.optim.Adam, loss='bce', lr=0.001)
    model.fit(X, y, batch_size=1024, steps_eval=10)
