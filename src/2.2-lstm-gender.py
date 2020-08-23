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
import torch.nn.utils.rnn as rnn_utils
from lstm import dataset
from lstm import model

data_path = "data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN=90
BATCH_SIZE = 512
HIDDEN_SIZE=512
N_EPOCHS=50
NUN_CLASS = 2
ATTN_SIZE=256
DROP=0.4
'''
adv_pro 77096
creative_id 3412772
ad_id 3027360
advertiser_id 57870
prodct_id 39057
product_category 18
industry 332
'''

#(4*n^2+3)/(n^2-8)
COLS=["advertiser_id","product_id","creative_id","industry"]#product_category","industry"]
VOCAB_SIZE= [57870+1,39057+1,3412772+1,332+1]#18+1,332+1]
EMBED_DIM = [256,256,64,256]
EMBED_SIZE=list(zip(VOCAB_SIZE,EMBED_DIM))
FN="" #预训练模型
#https://www.researchgate.net/post/How_to_choose_size_of_hidden_layer_and_number_of_layers_in_an_encoder-decoder_RNN
#HIDDEN_SIZE=sum(EMBED_DIM)
#HIDDEN_SIZE=int((4*HIDDEN_SIZE**2+3)/(HIDDEN_SIZE**2-8))

TARGET="gender"

def train_epoc(trn,model,criterion,optimizer,scheduler):

    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    
    for  label,data_x,data_len in tqdm(trn):
        #print("label",label)
        #print("data_x",data_x)
        #print("data_len",data_len)
        optimizer.zero_grad()
        label=label.to(device)
        data_len=data_len.to(device)   
        for i in range(len(data_x)):
            #data_x[i]=rnn_utils.pack_padded_sequence(data_x[i].to(device),data_len,batch_first=True,enforce_sorted=False)
            data_x[i]=data_x[i].to(device)
        output = model(data_x,data_len)
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()
        

    # Adjust the learning rate train_acc
    scheduler.step()

    return train_loss , train_acc 
def val_epoc(tst,model,criterion):
    model.eval()
    loss = 0
    acc = 0
    for label,data_x,data_len in tst:
        label=label.to(device)
        data_len=data_len.to(device) 
        for i in range(len(data_x)):
            #data_x[i]=rnn_utils.pack_padded_sequence(data_x[i].to(device),data_len,batch_first=True,enforce_sorted=False)
            data_x[i]=data_x[i].to(device) 
        with torch.no_grad():
            output = model(data_x,data_len)
            loss = criterion(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()

    return loss , acc 

if __name__ == "__main__":
    argv=sys.argv
    #EMBED_DIM = int(argv[1])
    #SEQ_LEN=int(argv[2])
    #MODE=argv[3]
    print("SEQ_LEN",SEQ_LEN)
    print("EMBED_SIZE",EMBED_SIZE)
    print("HIDDEN_SIZE",HIDDEN_SIZE)
    print("HIDDEN_SIZE",HIDDEN_SIZE)
    print("ATTN_SIZE",ATTN_SIZE)
    print("DROP",DROP)

    train_df=pd.read_pickle(data_path+"seq-train-lt90.pkl")
    #只训练一个
    #train_df=train_df[train_df["gender"]==1]
    la=["age","gender","demand"]
    train_df["age"]=train_df["age"]-1
    train_df["gender"]=train_df["gender"]-1
    for l in la:
        print(l,train_df[l].unique())
    val_df=train_df.sample(frac=0.1,random_state=999999)
    train_df=train_df.drop(val_df.index)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    #dataset
    train_ds=dataset.seq_dataset(train_df,seq_len=SEQ_LEN,cols_seq=COLS, col_y=TARGET,)
    val_ds=dataset.seq_dataset(val_df,seq_len=SEQ_LEN,cols_seq=COLS,col_y=TARGET,)
    
    model = model.baseline_model(EMBED_SIZE,HIDDEN_SIZE,SEQ_LEN,ATTN_SIZE, NUN_CLASS,device,DROP,bidirectional=True,num_layers=2).to(device)
    print(model)
    if os.path.exists(FN):
        print("load model")
        model.load_state_dict(torch.load(FN))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)#Adam 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.7)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=0)
    len_train=len(train_ds) 
    len_val=len(val_ds)  
    loss_plt=[]   
    for epoch in range(N_EPOCHS):
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=train_ds.generate_batch,num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                      collate_fn=val_ds.generate_batch)  
        train_loss, train_acc = train_epoc(train_dl,model,criterion,optimizer,scheduler)
        valid_loss, valid_acc = val_epoc(val_dl,model,criterion)
        print("epoch",epoch)
        print(f'\tLoss: {train_loss/len_train:.8f}(train)\t|\tAcc: {train_acc/len_train * 100:.3f}%(train)')
        print(f'\tLoss: {valid_loss/len_val:.8f}(valid)\t|\tAcc: {valid_acc/len_val * 100:.3f}%(valid)')
        loss_plt.append(train_acc/len_train)
        torch.save(model.state_dict(),TARGET+"-"+str(epoch)+".pth")
    print(loss_plt)
    plt.plot(range(len(loss_plt)), loss_plt) 
    plt.show()



