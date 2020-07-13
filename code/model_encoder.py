import torch
import torch.nn as nn
import transformers

from config import get_info_transfomer

"""
encoder_key: (Lstm | Transformers)
Model: <encoder_key>Layer(**input):
    input: x: torch.tensor[bsz, max_len]
    output: emb_ctx: torch.tensor[bsz, max_len, 2*d_ctx]
"""


class LstmLayer(nn.Module):
    def __init__(self, n_layer, d_ctx, d_emb, dropout=0,
                 if_freeze=False, n_vocab=None, embedding=None):
        super(LstmLayer, self).__init__()

        if embedding is None:
            self.embed = nn.Embedding(n_vocab, d_emb)
        else:
            self.embed = nn.Embedding.from_pretrained(embedding.clone(), freeze=if_freeze)
        self.encoder = nn.LSTM(d_emb, d_ctx, n_layer, dropout=dropout,
                               bidirectional=True, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)  # x[bsz, max_len] -> emb[bsz, max_len, dim_emb]
        emb_ctx, _ = self.encoder(emb)  # emb_ctx[bsz, max_len, 2*d_ctx]
        return emb_ctx


class TransformersLayer(nn.Module):
    def __init__(self, key_pretrained, d_ctx=64, dropout=0):
        super(TransformersLayer, self).__init__()
        info = get_info_transfomer(key_pretrained)
        key_architecture = info['architecture']
        d_hidden = info['d_hidden']
        key_model = key_architecture + "Model"
        self.encoder = getattr(transformers, key_model).from_pretrained(key_pretrained)
        self.finetune = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2 * d_ctx),
            nn.Tanh(),
            nn.Linear(2 * d_ctx, 2 * d_ctx),
        )

    def forward(self, x):
        with torch.no_grad():
            emb, _ = self.encoder(x)
        emb_ctx = self.finetune(emb)
        return emb_ctx


"""
Aggregate_key: (SelfAtt | DimAtt)
Model: <Aggregate_key>Layer(*input)
    input: emb_ctx, torch.tensor[bsz, T, 2*d_ctx]
    output: emb_sentence, torch.tensor[bsz, 2*d_ctx]
"""


class SelfAttLayer(nn.Module):
    def __init__(self, d_ctx, d_att):
        super(SelfAttLayer, self).__init__()
        self.layer_att = nn.Sequential(
            nn.Linear(2 * d_ctx, d_att),
            nn.Tanh(),
            nn.Linear(d_att, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, emb_ctx):  # [bsz, T, 2*d_ctx]
        attention = self.layer_att(emb_ctx)  # [bsz, T, 1]
        emb_att = torch.mul(emb_ctx, attention)  # [bsz, T, 2*d_ctx]
        emb_aggregate = torch.sum(emb_att, dim=1)  # [bsz, 2*d_ctx]
        return emb_aggregate

# class LSTMEncoder(nn.Module):
#     def __init__(self, d_emb, d_hidden, n_layer, d_a, feat_dim, num_classes,
#                  n_vocab=None, dropout_lstm=0, dropout_fc=0, embedding=None,
#                  key_pretrained='bert-base-chinese', sep_outlier=False):
#         super(LSTMEncoder, self).__init__()
#         self.encoder = encoder.LstmLayer(n_layer, d_hidden, d_emb, dropout_lstm,
#                                          n_vocab=n_vocab, embedding=embedding)
#         # self.encoder = encoder.TransformersLayer(key_pretrained, d_ctx=d_hidden)
#         self.aggregation = encoder.SelfAttLayer(d_hidden, d_a)
#         if sep_outlier:
#             self.reduction = nn.Sequential(
#                 nn.Dropout(dropout_fc),
#                 nn.Linear(2 * d_hidden, feat_dim)
#             )
#         else:
#             self.reduction = nn.Sequential(
#                 nn.Dropout(dropout_fc),
#                 nn.Linear(2 * d_hidden, num_classes)
#                 # nn.Linear(feat_dim, num_classes)
#             )
#
#     def forward(self, x):
#         emb_ctx = self.encoder(x)  # emb_ctx[bsz, max_len, d_ctx*2]
#         emb_aggregate = self.aggregation(emb_ctx)  # [bsz, 2*d_ctx]
#         emb_reduction = self.reduction(emb_aggregate)  # [bsz, dim_feature]
#         return emb_reduction

