import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.checkpoint import checkpoint
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TeModel(nn.Module):

    def __init__(self, ninp, seq_len,nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout,max_len=seq_len)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

        

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



    def forward(self, src):
        #SEQ,BATCH
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, self.src_mask)
        return output
class t_model(nn.Module):
    def __init__(self, emb_dims,trans_para ,lstm_para,num_class,deivce,seq_len=90,pool_size=5,attn_size=128):
        super().__init__()
        self.trans_para=trans_para
        self.lstm_para=lstm_para
        self.seq_len=seq_len
        self.no_of_embs=emb_dims
        self.pool_size=pool_size
        self.trans_layers=TeModel(self.no_of_embs, seq_len,**trans_para)
        self.seq_len=math.floor(seq_len/self.pool_size)
        self.attn_size=attn_size
        lstm_in=math.floor(self.no_of_embs/self.pool_size)
        self.lstm=nn.LSTM(lstm_in,batch_first=True,**lstm_para)
        layer_size=lstm_para["hidden_size"]
        if lstm_para["bidirectional"]:
            layer_size=layer_size*2
        self.pool=nn.MaxPool2d(self.pool_size)
        self.mlp=nn.Sequential(
            BasicLinearBlock(layer_size,512),
            BasicLinearBlock(512,256),
            nn.Linear(256,num_class)
        )

        #self._make_attn(deivce)
        #nn.init.kaiming_normal_(self.age.weight.data)
        #nn.init.kaiming_normal_(self.gender.weight.data)


    def forward(self,data,data_len):
        out = torch.cat(data, 2)
        out=out.permute(1,0,2)
        
        out=self.trans_layers(out)
        
        out=out.permute(1,0,2)
        hidden = self._initHidden(out.shape[0],out.device)
        #batch_out_pack = rnn_utils.pack_padded_sequence(out,data_len, batch_first=True,enforce_sorted=False)
        out=self.pool(out)
        out,hidden =self.lstm(out,hidden)
        #out = self.attention_net(out)
        #print(out.shape)
        #exit()
        out=out[:,-1,:]

        #out=self.mlp(out)
        out= self.mlp(out)
        return out
    def _initHidden(self,batch_size,device):
        h=torch.zeros(self.lstm_para["num_layers"],batch_size,self.lstm_para["hidden_size"])
        c=torch.zeros(self.lstm_para["num_layers"],batch_size,self.lstm_para["hidden_size"])
        if  self.lstm.bidirectional:
            h=torch.cat([h,h],dim=0)
            c=torch.cat([c,c],dim=0)
        return (h.to(device),c.to(device)) 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=90):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
 
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BasicLinearBlock(nn.Module):
    def __init__(self, in_size, out_size, drop_size=0):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_size)
        self.linear = nn.Linear(in_size, out_size)
        nn.init.kaiming_normal_(self.linear.weight.data)
        self.drop = nn.Dropout(drop_size)

    def forward(self, x):
        out = F.relu(self.linear(x))
        out = self.bn(out)
        out = self.drop(out)
        return out 