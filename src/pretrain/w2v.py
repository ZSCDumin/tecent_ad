# 序列进行预训练
import pandas as pd
import numpy as np
from gensim.models import Word2Vec,KeyedVectors
from datetime import datetime
import random
import multiprocessing
import os
import gc


class Item2Vec(object):
    def __init__(self, sentences, shuffle=True, output="output", model_name=None):
        super().__init__()
        self.output = output
        self.model_name = model_name
        self.shuffle = shuffle
        self.sentences = sentences
        if not isinstance(sentences[0][0], str):
            self.sentences = [self.to_str(sentence) for sentence in sentences]

    def to_str(self, sentence):
        return [str(word) for word in sentence]

    def train(self, emb_dim=64, window=5, negative=5,
              min_count=0, epochs=20, sg=1, hs=0,
              workers=multiprocessing.cpu_count()):
        """ shuffle : True for item2vec, False for word2vec
            sg: 1 for skip-grams 0 for cbow
            emb_dim: dimension of embedding """

        start = datetime.now()

        if not self.shuffle:
            self.w2v_model = Word2Vec(
                sentences=self.sentences,
                iter=epochs,
                size=emb_dim,
                window=window,
                negative=negative,
                sg=sg,
                hs=hs,
                workers=workers,
                min_count=min_count,)
            loss = self.w2v_model.get_latest_training_loss()
            print('Word2Vec Training Time: {},loss: {}'.format(
                datetime.now() - start, loss))
            if self.model_name is not None:
                self.w2v_model.save(
                    self.output + self.model_name + ".w2v_model")
            return self.w2v_model

        self.w2v_model = Word2Vec(
            sentences=self.sentences,
            iter=1,
            size=emb_dim,
            window=window,
            negative=negative,
            sg=sg,
            hs=hs,
            workers=workers,
            min_count=min_count)

        print("epoch 1 time: {}".format(datetime.now()-start))
        for i in range(2, epochs+1):
            for sentence in self.sentences:
                random.shuffle(sentence)
            start = datetime.now()
            self.w2v_model.train(
                self.sentences, total_examples=len(self.sentences), epochs=1)
            loss = self.w2v_model.get_latest_training_loss()
            print('epoch {} time: {} loss: {}'.format(
                i + 2, datetime.now() - start, loss))
        if self.model_name is not None:
            self.w2v_model.save(self.output + self.model_name + ".w2v_model")
        return self.w2v_model


def custom_split(str_in):
    arr = str_in.split(',')
    return list(set(arr))


if __name__ == "__main__":
    print("cpu",multiprocessing.cpu_count())
    seq_all = pd.read_pickle('data/' + "seq-all.pkl")
    cols = [ "product_category","ad_id", "advertiser_id", "creative_id","product_id","industry","adv_pro"]
    cols=["advertiser_id","product_id","ad_id","creative_id", "adv_pro"]
    dim=512
    print("cols",cols)
    ''' '''
    for col in cols:
        print(col)
        sentences =[custom_split(sentence) for sentence in seq_all[col]]
        model = Item2Vec(sentences, shuffle=True, output="./data/w2v/", model_name=col).train(emb_dim=dim, window=5, negative=6, epochs=20, workers=16)
        #model=Word2Vec.load('data/w2v/'+col+".w2v_model")
        model.wv.save('data/w2v/'+col+".wv")
        #break

    #cols = ["product_category"]
    exit()
    for col in cols:
        print("df",col)
        wv=KeyedVectors.load('data/w2v/'+col+".wv")
        df=pd.DataFrame(columns=(col,col+"_emb"))
        i=0
        for word in wv.vocab:
            df.loc[i] = [word,wv[word]]
            i=i+1
        df.to_pickle('data/w2v/'+col+"-"+str(dim)+".pkl")

