'''
基本模型的dataset
'''
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
#https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/2
def get_emb_dims(data,categorical_features):
    dims=[int(data[col].nunique()) for col in categorical_features]
    # from fastai
    emb_dims = [(x, min(600, round(1.6 * x**0.56))) for x in dims]
    return emb_dims
class seq_dataset(Dataset):
    def __init__(self, data,seq_len=200,col_y="age"):
        # 数据集个数
        self.n = data.shape[0]
        #序列
        self.seq_x = data["ap"].str.split(",")
        #实际值
        self.y=data[col_y].astype(np.int64).values
        self.seq=seq_len
    def __len__(self):
        return self.n
    def __getitem__(self,idx):
        x=np.array(self.seq_x[idx]).astype(int)
        if x.shape[0]>self.seq:
            x=x[:self.seq]     
        return [self.y[idx],x]
    #整理批次数据 embedingbag
    def generate_batch(self,batch):
        label = [entry[0] for entry in batch]

        text = [torch.tensor(entry[1]) for entry in batch]

        offsets = [0] + [len(entry) for entry in text]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, torch.tensor(label)
class seq_dataset1(Dataset):
    def __init__(self, data,seq_len=200,col_y="age"):
        # 数据集个数
        self.n = data.shape[0]
        #序列 

        self.seq_ap = data["adv_pro"].str.split(",")
        self.seq_cid = data["creative_id"].str.split(",")


        #实际值
        self.y=data[col_y].astype(np.int64).values
        self.seq=seq_len
   
    def __len__(self):
        return self.n
    def __getitem__(self,idx):

        ap=np.array(self.seq_ap[idx]).astype(int)
        if ap.shape[0]>self.seq:
            ap=ap[:self.seq]
        cid=np.array(self.seq_cid[idx]).astype(int)
        if cid.shape[0]>self.seq:
            cid=cid[:self.seq]       
        return [self.y[idx],ap,cid]
    #整理批次数据 embedingbag
    def generate_batch(self,batch):
        label = [entry[0] for entry in batch]

        text_ap = [torch.tensor(entry[1]) for entry in batch]
        offsets_ap = [0] + [len(entry) for entry in text_ap]
        offsets_ap = torch.tensor(offsets_ap[:-1]).cumsum(dim=0)
        text_ap = torch.cat(text_ap)

        text_cid = [torch.tensor(entry[2]) for entry in batch]
        offsets_cid = [0] + [len(entry) for entry in text_cid]
        offsets_cid = torch.tensor(offsets_cid[:-1]).cumsum(dim=0)
        text_cid = torch.cat(text_cid)
        return text_ap, offsets_ap, text_cid,offsets_cid,torch.tensor(label)