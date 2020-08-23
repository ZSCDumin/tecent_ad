import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
class baseline_model(nn.Module):
    def __init__(self, emb_dims, hidden_size,seq_len,attn_size,num_class,device,dropout=0.,bidirectional=False,num_layers=1):
        super().__init__()
        self.no_of_embs = emb_dims
        #self.emb_drop=nn.Dropout2d(0)
        self.lstm=nn.LSTM(self.no_of_embs,num_layers=num_layers,hidden_size=hidden_size,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        #self.bn=nn.BatchNorm1d(hidden_size)
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        if bidirectional:
            hidden_size=hidden_size*2
            #self.num_layers=self.num_layers*2
        
        self.output_layer =  nn.Sequential(
            BasicLinearBlock(hidden_size,512),
            BasicLinearBlock(512,256),
            nn.Linear(256,num_class)
        )
        '''
       
        self.output_layer=nn.Linear(hidden_size,num_class)
         '''


        #nn.init.kaiming_normal_(self.output_layer.weight.data)
        
        
        self.device=device
        self.bidirectional=bidirectional
        self.seq_len=seq_len
        self.attn_size=attn_size
        #attn
        self._make_attn()
    def forward(self,data,data_len):
        #print("data",data)
        #print("data_len",data_len)
        '''
        out=[]
        for i,emb_layer in enumerate(self.emb_layers):
            print(data[i])
            continue
            d=torch.tensor( data[i]).to(self.device)
            print (d.shape)
            out.append(emb_layer(data[i]))
            print(out[i].shape)
        
        exit()
        '''
        out = torch.cat(data, 2)

        
        #out = self.emb_drop(out)

        batch_out_pack = rnn_utils.pack_padded_sequence(out,
                                                  data_len, batch_first=True,enforce_sorted=False)
    
        #lstm
        hidden = self._initHidden(len(data[0]))
        lstm_out,_ =self.lstm(batch_out_pack,hidden)
        out_pad, _= rnn_utils.pad_packed_sequence(lstm_out, batch_first=True,total_length=self.seq_len)
        #out=torch.cat(out_pad,dim=2)
        attn_output = self.attention_net(out_pad)

        #idx = (data_len - 1).view(-1, 1).expand(len(data_len), out_pad.size(2))
        #time_dimension = 1 if batch_first else 0
        #time_dimension = 1
        #idx = idx.unsqueeze(1)

        #print(idx.shape)
        #双向 *2
        #last_output = out_pad.gather(time_dimension, idx).squeeze(time_dimension)

        
#https://blog.nelsonliu.me/2018/01/25/extracting-last-timestep-outputs-from-pytorch-rnns/
        #隐藏状态的最后一层就是输出的最后一层
        #print(hidden[0][-1])
        #exit()
        out=attn_output
        #print(out)
        return self.output_layer(out)
    def _initHidden(self,batch_size):
        h=torch.zeros(self.num_layers,batch_size,self.hidden_size)
        c=torch.zeros(self.num_layers,batch_size,self.hidden_size)
        if  self.lstm.bidirectional:
            h=torch.cat([h,h],dim=0)
            c=torch.cat([c,c],dim=0)
        return (h.to(self.device),c.to(self.device))
    def _make_attn(self):
        #layer_size=self.num_layers
        
        if self.bidirectional:
            hidden_size=self.hidden_size*2
            #pass
        self.w_omega = torch.zeros(hidden_size, self.attn_size).to(self.device)
        self.u_omega = torch.zeros(self.attn_size).to(self.device)
    def attention_net(self, lstm_output):
        #print("w_omega",self.w_omega.shape)
        #print("u_omega",self.u_omega.shape)
        if self.bidirectional:
            hidden_size=self.hidden_size*2
        #https://github.com/u784799i/biLSTM_attn/blob/master/model.py
        lstm_output=lstm_output.permute(1, 0, 2)
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        #print("lstm_output",lstm_output.shape)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, hidden_size])
        #print(output_reshape.shape)
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        #print("attn_tanh",attn_tanh.shape)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print("attn_hidden_layer",attn_hidden_layer.shape)
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.seq_len]) #长度
        #print(exps.size()) = (batch_size, squence_length)
        #print("exps",exps.shape)
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print("alphas",alphas.shape)
        #print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.seq_len, 1]) #长度
        #print("alphas_reshape",alphas_reshape.shape)
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        
        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        #print("attn_output",attn_output.shape)
        return attn_output


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