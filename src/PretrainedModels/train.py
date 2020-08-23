import sys
sys.path.append('..')
from deepmodels.models import Item2Vec
import pandas as pd
import multiprocessing
from Configs import _C as cfg


seq_all = pd.read_pickle('D:/_data/tencent2020/output/features/' + "seq-all.pkl")
sentences =[sentence.split(',') for sentence in seq_all["ad_id"]]

ad_id_model = Item2Vec(
    sentences, shuffle=False, save_dir="D:/", model_name="ad_id"
).train(emb_dim=128, window=5, negative=6, epochs=20, )

wv_df = pd.DataFrame()
for word in ad_id_model.wv.vocab:	
    wv_df[word] = ad_id_model.wv[word]

cols = wv_df.columns.to_numpy().astype('int32')
cols.sort()
cols = cols.astype(str).tolist()

wv_df[cols].T.to_pickle("./ad_id.wv")
