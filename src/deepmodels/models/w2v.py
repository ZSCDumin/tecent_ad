import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from datetime import datetime
import random, multiprocessing, os, gc
from tqdm import tqdm


class Item2Vec(object):
    def __init__(self, sentences, shuffle=True, save_dir = "./", model_name=None):
        """ shuffle: True for item2vec, False for word2vec """
        super().__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.shuffle = shuffle
        self.sentences = sentences
        if not isinstance(sentences[0][0], str):
            self.sentences = [to_str(sentence) for sentence in sentences]
            del(sentences)

    def train(self, emb_dim=64, window=5, negative=5, 
              min_count=0, epochs=20, sg=1, hs=0, 
              workers=multiprocessing.cpu_count()):
        """ shuffle : True for item2vec, False for word2vec
            sg: 1 for skip-grams, 0 for cbow
            emb_dim: dimension of embedding """

        start = datetime.now()

        if not self.shuffle:
            self.w2v_model = Word2Vec(
                sentences=self.sentences,
                iter=epochs, 
                size = emb_dim,
                window=window,
                negative=negative,
                sg=sg,
                hs=hs,
                workers=workers,
                min_count=min_count,)
            print('Word2Vec Training Time: {}'.format(datetime.now() - start))
            if self.model_name is not None:
                self.w2v_model.save(self.save_dir + "{}.w2v_model".format(self.model_name))
            return self.w2v_model

        self.w2v_model = Word2Vec(
            sentences=self.sentences, 
            iter=1,
            size = emb_dim,
            window=window,
            negative=negative,
            sg = sg,
            hs=hs,
            workers=workers,
            min_count=min_count)

        print("epoch 1 time: {}".format(datetime.now()-start))
        for i in range(2, epochs+1):
            for sentence in self.sentences:
                random.shuffle(sentence)
            start = datetime.now()
            self.w2v_model.train(self.sentences, total_examples=len(self.sentences), epochs=1)
            print('epoch {} time: {}'.format(i + 2, datetime.now() - start))
        if self.model_name is not None:
            self.w2v_model.save(self.save_dir + self.model_name + ".w2v_model")
        return self.w2v_model


def to_str(sentence):
    return [str(word) for word in sentence]
