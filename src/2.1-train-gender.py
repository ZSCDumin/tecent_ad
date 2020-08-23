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
EMBED_DIM = [256,64]
SEQ_LEN=150
BATCH_SIZE = 512
N_EPOCHS=10
MODE="max" #mean #max,sum
NUN_CLASS = 2
VOCAB_SIZE= [77096,3412772]

def train_epoc(trn,model,criterion,optimizer,scheduler):

    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    for  text_ap, offsets_ap, text_cid,offsets_cid, clss in tqdm(trn):
        optimizer.zero_grad()
        text_ap, text_cid=text_ap.long().to(device),text_cid.long().to(device)
        offsets_ap, offsets_cid, clss = offsets_ap.to(device),offsets_cid.to(device), clss.to(device)
        output = model(text_ap, offsets_ap,text_cid, offsets_cid)
        loss = criterion(output, clss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == clss).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss , train_acc 
def val_epoc(tst,model,criterion):
    model.eval()
    loss = 0
    acc = 0
    for text_ap, offsets_ap, text_cid,offsets_cid, clss in tst:
        text_ap, text_cid=text_ap.long().to(device),text_cid.long().to(device)
        offsets_ap, offsets_cid, clss = offsets_ap.to(device),offsets_cid.to(device), clss.to(device)
        with torch.no_grad():
            output = model(text_ap, offsets_ap,text_cid, offsets_cid)
            loss = criterion(output, clss)
            loss += loss.item()
            acc += (output.argmax(1) == clss).sum().item()

    return loss , acc 

if __name__ == "__main__":
    argv=sys.argv
    #EMBED_DIM = int(argv[1])
    #SEQ_LEN=int(argv[2])
    #MODE=argv[3]
    print("EMBED_DIM",EMBED_DIM)
    print("SEQ_LEN",SEQ_LEN)
    print("MODE",MODE)
    print("VOCAB_SIZE",VOCAB_SIZE)
    train_df=pd.read_pickle(data_path+"seq-train.pkl")
    la=["age","gender","demand"]
    train_df["age"]=train_df["age"]-1
    train_df["gender"]=train_df["gender"]-1
    for l in la:
        print(l,train_df[l].unique())
    val_df=train_df.sample(frac=0.2)#,random_state=999999
    train_df=train_df.drop(val_df.index)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    train_ds=dataset.seq_dataset1(train_df,seq_len=SEQ_LEN,col_y="gender",)

    val_ds=dataset.seq_dataset1(val_df,seq_len=SEQ_LEN,col_y="gender",)
    
    model = model.baseline_model1(VOCAB_SIZE[0], EMBED_DIM[0], VOCAB_SIZE[1], EMBED_DIM[1],NUN_CLASS,MODE).to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)#Adam 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.7)
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
        loss_plt.append(train_acc/len_train)
        torch.save(model.state_dict(),"gender-"+str(epoch)+".pth")
    print(loss_plt)
    plt.plot(range(len(loss_plt)), loss_plt) 
    plt.show()



