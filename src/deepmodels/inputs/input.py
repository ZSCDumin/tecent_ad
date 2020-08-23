from torch import nn
from .featrure_column import *


def create_embedding_dict(feature_columns, init_std=0.0001, linear=False, sparse=False, device="cpu"):
    
    sparse_feature_columns = [fc for fc in feature_columns if isinstance(fc, SparseFeat)]
    varlen_sparse_feature_columns = [fc for fc in feature_columns if isinstance(fc, VarLenSparseFeat)]

    embedding_dict = nn.ModuleDict({
        fc.embedding_name: nn.Embedding(
            fc.vocabulary_size, 
            fc.embedding_dim if not linear else 1, 
            sparse=sparse, padding_idx=0,
        ) for fc in sparse_feature_columns + varlen_sparse_feature_columns
    })

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)

