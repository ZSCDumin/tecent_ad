import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TransformerModel(nn.Module):
    '''
    embed_size ：嵌入参数 ntoken,ninp
    nhead ：the number of heads in the multiheadattention models
    nhid：the dimension of the feedforward network model
    nlayers：the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout：
    '''
    def __init__(self, ntoken,ninp, seq_len,nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout,max_len=seq_len)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
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

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, self.src_mask)
        return output
   

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

class t_model(nn.Module):
    def __init__(self, emb_dims,trans_para ,lstm_para,num_class,seq_len=90):
        super().__init__()
        self.trans_para=trans_para
        self.lstm_para=lstm_para
        self.seq_len=seq_len
        self.no_of_embs = sum([y for x, y in emb_dims]) if  emb_dims else 0
        #使用transformer替代embeding
        self.trans_layers = nn.ModuleList([TransformerModel(x, y,seq_len,**trans_para) for x, y in emb_dims] ) if  emb_dims else None 
        self.lstm=nn.LSTM(self.no_of_embs,batch_first=True,**lstm_para)
        layer_size=lstm_para["hidden_size"]
        if lstm_para["bidirectional"]:
            layer_size=layer_size*2
        self.output_layer =  nn.Linear(layer_size,num_class)
        nn.init.kaiming_normal_(self.output_layer.weight.data)


    def forward(self,data,data_len):
        out = [trans_layers(data[i].permute(1,0)) for i,trans_layers in enumerate(self.trans_layers)]
        out = torch.cat(out, 2)
        out=out.permute(1,0,2)
        hidden = self._initHidden(out.shape[0],out.device)
        #batch_out_pack = rnn_utils.pack_padded_sequence(out,data_len, batch_first=True,enforce_sorted=False)
        out,hidden =self.lstm(out,hidden)
        out=out[:,-1,:]

        out=self.output_layer(out)
        return out
    def _initHidden(self,batch_size,device):
        h=torch.zeros(self.lstm_para["num_layers"],batch_size,self.lstm_para["hidden_size"])
        c=torch.zeros(self.lstm_para["num_layers"],batch_size,self.lstm_para["hidden_size"])
        if  self.lstm.bidirectional:
            h=torch.cat([h,h],dim=0)
            c=torch.cat([c,c],dim=0)
        return (h.to(device),c.to(device))        


#################################################
#整合
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

class t1_model(nn.Module):
    def __init__(self, emb_dims,trans_para ,lstm_para,num_class,seq_len=90):
        super().__init__()
        self.trans_para=trans_para
        self.lstm_para=lstm_para
        self.seq_len=seq_len
        self.no_of_embs = sum([y for x, y in emb_dims]) if  emb_dims else 0
        #使用transformer替代embeding
        #self.trans_layers = nn.ModuleList([TransformerModel(x, y,seq_len,**trans_para) for x, y in emb_dims] ) if  emb_dims else None 
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y,padding_idx=0) for x, y in emb_dims] ) if  emb_dims else None
        self.trans_layers=TeModel(self.no_of_embs, seq_len,**trans_para)
        self.lstm=nn.LSTM(self.no_of_embs,batch_first=True,**lstm_para)
        layer_size=lstm_para["hidden_size"]
        if lstm_para["bidirectional"]:
            layer_size=layer_size*2
        self.output_layer =  nn.Linear(layer_size,num_class)
        nn.init.kaiming_normal_(self.output_layer.weight.data)


    def forward(self,data,data_len):
        out = [emb_layers(data[i].permute(1,0)) for i,emb_layers in enumerate(self.emb_layers)]
        out = torch.cat(out, 2)
        out=self.trans_layers(out)
        out=out.permute(1,0,2)
        hidden = self._initHidden(out.shape[0],out.device)
        #batch_out_pack = rnn_utils.pack_padded_sequence(out,data_len, batch_first=True,enforce_sorted=False)
        out,hidden =self.lstm(out,hidden)
        out=out[:,-1,:]

        out=self.output_layer(out)
        return out
    def _initHidden(self,batch_size,device):
        h=torch.zeros(self.lstm_para["num_layers"],batch_size,self.lstm_para["hidden_size"])
        c=torch.zeros(self.lstm_para["num_layers"],batch_size,self.lstm_para["hidden_size"])
        if  self.lstm.bidirectional:
            h=torch.cat([h,h],dim=0)
            c=torch.cat([c,c],dim=0)
        return (h.to(device),c.to(device))  