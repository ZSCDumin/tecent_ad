'''
第二步:训练
读取处理好的数据，训练模型
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
import torch.nn.utils.rnn as rnn_utils
from lstm import dataset
from lstm import transformer

data_path = "data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
NUM_CLASS = 10
N_EPOCHS = 10
SEQ_LEN = 90
TF_PARAM = {}
TF_PARAM["nhid"] = 200
TF_PARAM["nhead"] = 2
TF_PARAM["nlayers"] = 2
TF_PARAM["dropout"] = 0.4

LSTM_PARAM = {}
LSTM_PARAM["num_layers"] = 2
LSTM_PARAM["hidden_size"] = 256
LSTM_PARAM["bidirectional"] = True
LSTM_PARAM["dropout"] = 0.4
ATTN_SIZE = 128

'''
adv_pro 77096
creative_id 3412772
ad_id 3027360
advertiser_id 57870
product_id 39057
product_category 18
industry 332
'''

# (4*n^2+3)/(n^2-8)"
# product_category","industry"]
COLS = ["advertiser_id", "product_id", "industry", "creative_id", "ad_id"]
VOCAB_SIZE = [57870+1, 39057+1, 332+1,3412772+1,3027360+1]  # 18+1,332+1]
EMBED_DIM = [128, 128, 128, 64, 64]
EMBED_SIZE = list(zip(VOCAB_SIZE, EMBED_DIM))
FN = "1"  # 预训练模型
# https://www.researchgate.net/post/How_to_choose_size_of_hidden_layer_and_number_of_layers_in_an_encoder-decoder_RNN
# HIDDEN_SIZE=sum(EMBED_DIM)
# HIDDEN_SIZE=int((4*HIDDEN_SIZE**2+3)/(HIDDEN_SIZE**2-8))


def train_epoc(trn, model, criterion, optimizer, scheduler):

    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    pbar = tqdm(trn)
    for label, data_x, data_len in pbar:
        # print("aa",pbar.n)
        # exit()
        # print("label",label)
        # print("data_x",data_x)
        # print("data_len",data_len)
        optimizer.zero_grad()
        label = label.to(device)
        data_len = data_len.to(device)
        for i in range(len(data_x)):
            # data_x[i]=rnn_utils.pack_padded_sequence(data_x[i].to(device),data_len,batch_first=True,enforce_sorted=False)
            data_x[i] = data_x[i].to(device)
        output = model(data_x,data_len)

        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()

        # Adjust the learning rate train_acc
    scheduler.step()

    return train_loss, train_acc


def val_epoc(tst, model, criterion):
    model.eval()
    loss = 0
    acc = 0
    for label, data_x, data_len in tst:
        label = label.to(device)
        data_len = data_len.to(device)
        for i in range(len(data_x)):
            # data_x[i]=rnn_utils.pack_padded_sequence(data_x[i].to(device),data_len,batch_first=True,enforce_sorted=False)
            data_x[i] = data_x[i].to(device)
        with torch.no_grad():
            output = model(data_x, data_len)
            loss = criterion(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()

    return loss, acc


if __name__ == "__main__":
    argv = sys.argv

    print("SEQ_LEN", SEQ_LEN)
    print("EMBED_SIZE", EMBED_SIZE)
    print("LSTM", LSTM_PARAM)
    print("TRANS", TF_PARAM)
    print("NUM_CLASS",NUM_CLASS)


    train_df = pd.read_pickle(data_path+"seq-train-lt90.pkl")
    # 只训练一个
    # train_df=train_df[train_df["gender"]==1]
    la = ["age", "gender", "demand"]
    train_df["age"] = train_df["age"]-1
    train_df["gender"] = train_df["gender"]-1
    for l in la:
        print(l, train_df[l].unique())
    val_df = train_df.sample(frac=0.1, random_state=999999)
    train_df = train_df.drop(val_df.index)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    # dataset
    train_ds = dataset.seq_dataset(
        train_df, seq_len=SEQ_LEN, cols_seq=COLS, col_y="age",)
    val_ds = dataset.seq_dataset(
        val_df, seq_len=SEQ_LEN, cols_seq=COLS, col_y="age",)

    model = transformer.t_model(emb_dims=EMBED_SIZE,
                                seq_len=SEQ_LEN,
                                trans_para=TF_PARAM,
                                lstm_para=LSTM_PARAM,
                                num_class=NUM_CLASS
                                ).to(device)
    print(model)
    if os.path.exists(FN):
        print("load model")
        model.load_state_dict(torch.load(FN))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Adam 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=0)
    len_train = len(train_ds)
    len_val = len(val_ds)
    loss_plt = []
    for epoch in range(N_EPOCHS):
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=train_ds.generate_batch)  # ,num_workers=2
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            collate_fn=val_ds.generate_batch)
        train_loss, train_acc = train_epoc(
            train_dl, model, criterion, optimizer, scheduler)
        valid_loss, valid_acc = val_epoc(val_dl, model, criterion)
        print("epoch", epoch)
        print(
            f'\tLoss: {train_loss/len_train:.8f}(train)\t|\tAcc: {train_acc/len_train * 100:.3f}%(train)')
        print(
            f'\tLoss: {valid_loss/len_val:.8f}(valid)\t|\tAcc: {valid_acc/len_val * 100:.3f}%(valid)')
        loss_plt.append(train_acc/len_train)
        torch.save(model.state_dict(), "age-"+str(epoch)+".pth")
    print(loss_plt)
    plt.plot(range(len(loss_plt)), loss_plt)
    plt.show()
