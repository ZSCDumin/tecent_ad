import time, pdb
from collections import deque
from tqdm import tqdm
from sklearn.metrics import accuracy_score, auc, roc_auc_score

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from ..inputs import create_embedding_dict, build_feature_index


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """ Construct activation layers

        Args:
            act_name: str or nn.Module, name of activation function
            hidden_size: int, used for Dice activation
            dice_dim: int, used for Dice activation
        Return:
            act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):

    def __init__(self, 
            inputs_dim,
            hidden_units=(16, 8),
            activation='relu', 
            l2_reg=0, 
            dropout_rate=0, 
            use_bn=False,
            init_std=0.0001, 
            dice_dim=3, seed=1024, device='cuda:0'):
        super(DNN, self).__init__()

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        if 0 < dropout_rate < 1:
            self.dropout = nn.Dropout(dropout_rate)
        
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList([
            activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(self.device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            if hasattr(self, 'dropout'):
                fc = self.dropout(fc)
            deep_input = fc

        return deep_input


class LocalActivationUnit(nn.Module):

    def __init__(self,
            hidden_units=(64, 32),
            embedding_dim=4,
            activation='sigmoid',
            dropout_rate=0,
            dice_dim=3, l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(
            inputs_dim=embedding_dim,
            hidden_units=hidden_units,
            activation=activation,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            dice_dim=dice_dim,
            use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, user_behavior):

        # _, user_behavior_len, _ = user_behavior.size()

        attention_input = user_behavior

        attention_output = self.dnn(attention_input)   # [B, T, 4*E] -> [B, T, H[-1]]

        attention_score = self.dense(attention_output) # [B, T, H[-1]] -> [B, T, 1]

        return attention_score


class AttentionSequencePoolingLayer(torch.nn.Module):

    def __init__(self, 
            embedding_dim=4,
            att_hidden_units=(64, 32), 
            att_activation='sigmoid', 
            weight_normalization=False,
            return_score=False, 
            supports_masking=False,  **kwargs):
        super().__init__()

        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(
            hidden_units=att_hidden_units, 
            embedding_dim=embedding_dim,
            activation=att_activation, 
            dropout_rate=0, use_bn=False
        )

    def forward(self, keys, *length_or_mask):

   


class User_statisic_att_model(torch.nn.Module):

    def __init__(self, feature_columns, # hidden_size=64, 
                 att_hidden_units=(64, 16), dnn_hidden_units=(16, 4), 
                 att_activation='relu', device="cuda:0"):
        super().__init__()
        self.feature_columns = feature_columns
        self.device = device if torch.cuda.is_available() else "cpu"

        self.embedding_dict = create_embedding_dict(
            feature_columns, init_std=0.0001, device=self.device)
        self.feature_index = build_feature_index(feature_columns)

        self.attention = AttentionSequencePoolingLayer(
            embedding_dim=sum(fc.embedding_dim for fc in self.feature_columns),
            att_hidden_units=att_hidden_units,
            activation=att_activation,
            return_score=False,
        )

        self.dnn = DNN(inputs_dim=sum(fc.embedding_dim for fc in self.feature_columns), 
                       hidden_units=dnn_hidden_units, device=self.device)
        self.out = nn.Linear(dnn_hidden_units[-1], 1, bias=True)
        self.to(self.device)

    def forward(self, *X):

        emb_list = [self.embedding_dict[fc.embedding_name](
            X[self.feature_index[fc.name][-1]]
        ) for fc in self.feature_columns]

        emb_input = torch.cat(emb_list, dim=-1) # -> [B, T, E]

        output_att = self.attention(emb_input, X[self.feature_index["seq_length"][-1]].long())

        output = self.dnn(output_att.squeeze())

        logit = self.out(output)

        return logit

    def get_emb_list(self, *X):

        lengths = X[self.feature_index["seq_length"][-1]].long()

        # 排序
        lengths_descending, idx_descending = lengths.sort(0, descending=True)
        _, un_idx = idx_descending.sort(dim=0)

        X_descending = [tensor[idx_descending] for tensor in X]

        emb_list = [self.embedding_dict[fc.embedding_name](
            X_descending[self.feature_index[fc.name][-1]]
        ) for fc in self.feature_columns]

        return emb_list, lengths_descending, un_idx

    def fit(self, X, y, validation_data=None, batch_size=256, epochs=20, 
            steps_eval=None, verbose=1, grad_accum=1, init_epoch=1):

        if isinstance(X, dict):
            # X = [X[feat] if isinstance(X[feat], torch.Tensor) else (
            #                 torch.from_numpy(X[feat]) if isinstance(X[feat], np.ndarray) 
            #                     else torch.tensor(X[feat]))
            #      for feat in self.feature_index]
            X = [X[feat] if isinstance(X[feat], torch.Tensor) else torch.from_numpy(
                        X[feat] if isinstance(X[feat], np.ndarray) else np.asarray(X[feat])
                    ).type(torch.long) for feat in self.feature_index]
        else:
            raise NotImplementedError

        X_val, y_val = None, None
        if validation_data is not None:
            X_val, y_val, *val_sample_weights = validation_data

        train_set = TensorDataset(*X, y)
        train_loader = DataLoader(train_set, batch_size=batch_size)

        valid_set = []
        if y_val is not None:
            valid_set = TensorDataset(*X_val, y_val)
            valid_loader = DataLoader(valid_set, batch_size=batch_size)

        sample_num = len(train_set)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_set), len(valid_set), steps_per_epoch))

        total_loss_epoch = 0.
        metricT = Metric(keep_size=steps_per_epoch)
        self.train()
        for epoch in range(init_epoch, epochs+init_epoch):

            start_time = time.time()
            with tqdm(enumerate(train_loader, start=1), initial=1, disable=verbose != 1) as it:
                for i, (*X_batch, y_batch) in it:

                    X_batch = [tensor.to(self.device) for tensor in X_batch]
                    y_batch = y_batch.float().to(self.device)

                    logits = self(*X_batch)
                    loss = self.loss_func(logits.squeeze(), y_batch)
                    if grad_accum > 1:
                        loss /= grad_accum
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    ############ 
                    metricT(logits.sigmoid().detach().squeeze().cpu().numpy(),
                            y_batch.squeeze().data.cpu().numpy(), loss.item())

                    if steps_eval and i % steps_eval == 0: # pdb.set_trace()
                        if y_val:
                            self.evaluate(X_val, y_val)
                    
                    if i % 1000 == 0:
                        print("Saving")
                        pdb.set_trace()

            it.close()

            epoch_time = int(time.time() - start_time)
            if verbose > 0:
                print('Epoch {0}/{1}'.format(epoch, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, total_loss_epoch / sample_num)

                print(eval_str)

    @torch.no_grad()
    def predict(self, X, batch_size=256,):
        if isinstance(X, dict):
            X = [X[feat] if isinstance(X[feat], torch.Tensor) else (
                            torch.from_numpy(X[feat]) if isinstance(X[feat], np.ndarray) 
                                else torch.tensor(X[feat]))
                 for feat in self.feature_index]
        
        tensor_set = TensorDataset(*X)
        data_loader = DataLoader(tensor_set, batch_size, shuffle=False)

        pred_ans = []
        for _, X_batch in enumerate(data_loader, start=1):
            X_batch = [tensor.to(self.device) for tensor in X_batch]
            y_pred = self(*X_batch).detach().sigmoid().squeeze().cpu().numpy()
            pred_ans.append(y_pred)
        
        return np.concatenate(pred_ans)

    @torch.no_grad()
    def evaluate(self, X, y, batch_size=256):
        pred_ans = self.predict(X, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def compile(self, optimizer, loss=None, metrics=None, **kwargs):

        if isinstance(optimizer, torch.optim.Optimizer):
            optim = optimizer
        elif issubclass(optimizer, torch.optim.Optimizer):
            optim = optimizer(self.parameters(), lr=kwargs['lr'])
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=kwargs['lr'])
        else:
            raise NotImplementedError

        if isinstance(loss, str):
            if loss.lower() == 'bce':
                loss_func = F.binary_cross_entropy_with_logits
        elif isinstance(loss, torch.nn.Module) or callable(loss) or hasattr(loss, "__call__"):
            loss_func = loss
        elif issubclass(loss, torch.nn.Module):
            loss_func = loss()
        else:
            raise NotImplementedError

        metrics_={}
        if metrics:
            for metric in metrics:
                if metric.lower() in ["accuracy", "acc"]:
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))

        optim.zero_grad()
        self.optimizer = optim
        self.loss_func = loss_func
        self.metrics = metrics_


class Metric(object):
    def __init__(self, win_size=8, keep_size=1000):
        super().__init__()
        self.deq = deque(maxlen=keep_size)
        self.win = deque([0], maxlen=win_size)
        self.win_size = win_size
        self.steps = 0

    def __call__(self, y_score, y_true, loss):
        self.deq.append(loss)
        self.win.append(loss)
        self.steps += 1
        if self.steps % 20 == 0:
            print("Iter [{}/__], ACC: {:.3f}, AUC: {:4f}, Loss Win: {:.4f}".format(
                self.steps, self.acc(y_score > .5, y_true), self.auc(y_score, y_true), np.mean(self.win)))

    def acc(self, y_pred, y_true):
        return accuracy_score(y_true, y_pred)
    
    def auc(self, y_score, y_true):
        return roc_auc_score(y_true, y_score)


"""
self.lstm = torch.nn.LSTM(
    sum(fc.embedding_dim for fc in feature_columns),
    hidden_size=hidden_size,
    # num_layers=num_layers,
    bias=True,
    batch_first=True,
)

def forward_(self, *X):

        emb_list, lengths_descending, un_idx = self.get_emb_list(*X)
        emb_input = torch.cat(emb_list, dim=-1) # -> [B, T, E]

        emb_packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input=emb_input, lengths=lengths_descending, batch_first=True)
        packed_out, _ = self.lstm(emb_packed_input)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True)
        output = torch.index_select(output, 0, un_idx)

        max_pooling, _ = torch.max(output, dim=-2) # ?

        output = self.dnn(max_pooling)

        logit = self.out(output)

        return logit
"""
