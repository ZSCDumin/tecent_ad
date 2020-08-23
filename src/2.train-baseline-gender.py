'''
第二步:训练
读取处理好的数据，训练模型
'''
import torch,os,time,sys
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
from baseline import dataset
from baseline import model

data_path = "data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 600
SEQ_LEN=200
BATCH_SIZE = 1024
N_EPOCHS=10
MODE="sum" #mean #max


def train_epoc(trn,model,criterion,optimizer,scheduler):

    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    for  text, offsets, clss in tqdm(trn):

        optimizer.zero_grad()
        text, offsets, clss = text.long().to(
            device), offsets.to(device), clss.to(device)
        output = model(text, offsets)
        loss = criterion(output, clss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == clss).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss , train_acc 
def val_epoc(tst,model,criterion):
    model.eval
    loss = 0
    acc = 0
    for text, offsets, clss in tst:
        text, offsets, clss = text.long().to(device), offsets.to(device), clss.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, clss)
            loss += loss.item()
            acc += (output.argmax(1) == clss).sum().item()

    return loss , acc 

if __name__ == "__main__":
    argv=sys.argv
    EMBED_DIM = int(argv[1])
    SEQ_LEN=int(argv[2])
    MODE=argv[3]
    print("EMBED_DIM",EMBED_DIM)
    print("SEQ_LEN",SEQ_LEN)
    print("MODE",MODE)
    NUN_CLASS = 2
    VOCAB_SIZE= 77096
    train_df=pd.read_pickle(data_path+"seq-train.pkl")
    la=["age","gender","demand"]
    train_df["age"]=train_df["age"]-1
    train_df["gender"]=train_df["gender"]-1
    for l in la:
        print(l,train_df[l].unique())
    val_df=train_df.sample(n=10000,random_state=999999)
    train_df=train_df.drop(val_df.index)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    train_ds=dataset.seq_dataset(train_df,SEQ_LEN,"gender")
    val_ds=dataset.seq_dataset(val_df,SEQ_LEN,"gender")
    model = model.baseline_model(VOCAB_SIZE, EMBED_DIM, NUN_CLASS,MODE).to(device)
    criterion = torch.nn.CrossEntropyLoss()#.to(device)
    optimizer = torch.optim.SGD(model.parameters(),weight_decay=1e-5,lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.75)
    len_train=len(train_ds) 
    len_val=len(val_ds)  
    loss_plt=[]   
    for epoch in range(N_EPOCHS):
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=train_ds.generate_batch)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                      collate_fn=val_ds.generate_batch)  
        train_loss, train_acc = train_epoc(train_dl,model,criterion,optimizer,scheduler)
        valid_loss, valid_acc = val_epoc(val_dl,model,criterion)
        print("epoch",epoch)
        print(f'\tLoss: {train_loss/len_train:.8f}(train)\t|\tAcc: {train_acc/len_train * 100:.3f}%(train)')
        print(f'\tLoss: {valid_loss/len_val:.8f}(valid)\t|\tAcc: {valid_acc/len_val * 100:.3f}%(valid)')
        torch.save(model.state_dict(),"gender"+str(epoch)+".pth")
        loss_plt.append(train_acc/len_train)
    print(loss_plt)
    plt.plot(range(len(loss_plt)), loss_plt) 
    plt.show()



