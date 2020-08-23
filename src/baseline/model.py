'''序列的基本模型
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
先跑起来，然后可以堆几层mlp测试
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class baseline_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class,mode="mean"):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode=mode)
        self.emb_drop=nn.Dropout(0.)
        self.fc1 = BasicLinearBlock(embed_dim, 256,0.)
        self.fc2 = BasicLinearBlock(256, 128,0.)
        self.fc=nn.Linear(128,num_class)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.embedding.weight.data)
        nn.init.kaiming_normal_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        x=self.emb_drop(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return self.fc(x)
class baseline_model1(nn.Module):
    def __init__(self, vs_ap, ed_ap,vs_cid,ed_cid, num_class,mode="mean"):
        super().__init__()
        self.embedding_ap = nn.EmbeddingBag(vs_ap, ed_ap, mode=mode)
        self.embedding_cid = nn.EmbeddingBag(vs_cid, ed_cid, mode=mode)
        self.emb_drop=nn.Dropout(0.)
        self.fc1 = BasicLinearBlock(ed_cid+ed_ap, 512,0.)
        self.fc2 = BasicLinearBlock(512, 256,0.)
        self.fc=nn.Linear(256,num_class)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.embedding_ap.weight.data)
        nn.init.kaiming_normal_(self.embedding_cid.weight.data)
        nn.init.kaiming_normal_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, text_ap, offsets_ap,text_cid, offsets_cid):
        emb_ap = self.embedding_ap(text_ap, offsets_ap)
        emb_cid=self.embedding_cid(text_cid, offsets_cid)
        out = torch.cat([emb_ap,emb_cid], 1)
        out = self.emb_drop(out)
        out=self.fc1(out)
        out=self.fc2(out)
        return self.fc(out)
# 基本模块，BN  线性 激活 

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