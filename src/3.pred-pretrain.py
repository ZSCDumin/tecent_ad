'''
第三步:预测
加载训练好的模型，生成提交数据
'''
import torch
import os
import time
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
import matplotlib.pyplot as plt
from pretrain import dataset
from pretrain import model as transformer

data_path = "data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
NUM_CLASS = 10
N_EPOCHS = 10
SEQ_LEN = 90
POOL_SIZE=5
TF_PARAM = {}
TF_PARAM["nhid"] =512
TF_PARAM["nhead"] = 8
TF_PARAM["nlayers"] = 6
TF_PARAM["dropout"] = 0.

LSTM_PARAM = {}
LSTM_PARAM["num_layers"] = 2
LSTM_PARAM["hidden_size"] = 512
LSTM_PARAM["bidirectional"] = True
LSTM_PARAM["dropout"] = 0.
ATTN_SIZE = 128


COLS = [ "advertiser_id","ad_id", ]
EMBED_SIZE = len(COLS)*512
print(EMBED_SIZE)

test_df = pd.read_pickle(data_path+"seq-test.pkl")
test_df["age"]=0
test_df["gender"]=0
test_df.reset_index(drop=True, inplace=True)
print(test_df.info())

NUN_CLASS = 10
model = transformer.t_model(emb_dims=EMBED_SIZE,
                                seq_len=SEQ_LEN,
                                trans_para=TF_PARAM,
                                lstm_para=LSTM_PARAM,
                                num_class=NUM_CLASS,
                                pool_size=POOL_SIZE,
                                attn_size=ATTN_SIZE,
                                deivce=device,
                                ).to(device)
model.load_state_dict(torch.load("pt141/pt512-age-9.pth"))
model = model.to(device)
model.eval()
test_ds = dataset.seq_dataset(test_df,wv_path="data/w2v/512/",seq_len=SEQ_LEN,cols_seq=COLS,)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,collate_fn=test_ds.generate_batch,)
pred=[]

for label,data_x,data_len in tqdm(test_dl):
    label=label.to(device)
    data_len=data_len.to(device) 
    for i in range(len(data_x)):
        data_x[i]=data_x[i].to(device) 

    with torch.no_grad():
        output = model(data_x,data_len)
        label=output.argmax(1)+1
        pred.append(label.cpu())

test_df["predicted_age"]=torch.cat(pred).numpy()
print("predicted_age:",test_df["predicted_age"].nunique())
NUM_CLASS = 2    
model = transformer.t_model(emb_dims=EMBED_SIZE,
                                seq_len=SEQ_LEN,
                                trans_para=TF_PARAM,
                                lstm_para=LSTM_PARAM,
                                num_class=NUM_CLASS,
                                pool_size=POOL_SIZE,
                                attn_size=ATTN_SIZE,
                                deivce=device,
                                ).to(device)
model.load_state_dict(torch.load("pt141/512-gender-9.pth"))
model = model.to(device)
model.eval()
#test_ds = dataset.seq_dataset(test_df, seq_len=SEQ_LEN,cols_seq=COLS, col_y="gender",)
#Stest_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,collate_fn=test_ds.generate_batch)
pred=[]
for label,data_x,data_len in tqdm(test_dl):
    label=label.to(device)
    data_len=data_len.to(device) 
    for i in range(len(data_x)):
        data_x[i]=data_x[i].to(device) 

    with torch.no_grad():
        output = model(data_x,data_len)
        label=output.argmax(1)+1
        pred.append(label.cpu())
test_df["predicted_gender"]=torch.cat(pred).numpy()
print("predicted_gender:",test_df["predicted_gender"].nunique()) 

submit=test_df[["user_id","predicted_age","predicted_gender"]]
submit.to_csv("pre-train-512-submit.csv",index=False)

        
        