'''
基本模型的dataset
'''
import pandas as pd
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
#https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/2
def get_emb_dims(data,categorical_features):
    dims=[int(data[col].nunique()) for col in categorical_features]
    # from fastai
    emb_dims = [(x, min(600, round(1.6 * x**0.56))) for x in dims]
    return emb_dims
class seq_dataset(Dataset):
    def __init__(self, data,seq_len,cols_seq,col_y="age"):
        # 数据集个数
        self.n = data.shape[0]
        #序列 
        self.seq_x={}
        for col in cols_seq:
           
            self.seq_x[col]=data[col].str.split(",")


        #实际值
        self.y=data[col_y].astype(np.int64).values
        #其他
        self.seq_len=seq_len
        self.seq_cols=cols_seq
 

   
    def __len__(self):
        return self.n
    def __getitem__(self,idx):
        cols_seq=[]
        for col in self.seq_cols:
            seq=np.array(self.seq_x[col][idx]).astype(int)
            if seq.shape[0]>self.seq_len:
                seq=seq[:self.seq_len]
            cols_seq.append(torch.tensor(seq))
     
        return [self.y[idx]]+cols_seq #list
    #整理批次数据
    #label [batch,Y]
    #seq [cols,batch,seq] list
    #seq_len 每个序列的长度
    def generate_batch(self,batch):
        #batch:[y,col1,col2,...]
        #batch.sort(key=lambda x: len(x), reverse=True)
        label = [entry[0] for entry in batch]
        seq=[]
        seq_len=[len(entry[1]) for entry in batch]
        for i in range(1,len(self.seq_cols)+1):
            x=[entry[i] for entry in batch]
            x=rnn_utils.pad_sequence(x,batch_first=True,padding_value=0).long()
            seq.append(x)
        return torch.tensor(label),seq,torch.tensor(seq_len)