import torch
import torch.nn as nn
import numpy as np
import math
import model_encoder as encoder
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, d_emb, d_hidden, n_layer, d_a, feat_dim, num_classes,
                 n_vocab=None, dropout_lstm=0, dropout_fc=0, embedding=None,
                 key_pretrained='bert-base-chinese', sep_outlier=False):
        super(LSTMEncoder, self).__init__()
        self.encoder = encoder.LstmLayer(n_layer, d_hidden, d_emb, dropout_lstm,
                                         n_vocab=n_vocab, embedding=embedding)
        # self.encoder = encoder.TransformersLayer(key_pretrained, d_ctx=d_hidden)
        self.aggregation = encoder.SelfAttLayer(d_hidden, d_a)
        if sep_outlier:
            self.reduction = nn.Sequential(
                nn.Dropout(dropout_fc),
                nn.Linear(2 * d_hidden, feat_dim),
                nn.Linear(feat_dim, feat_dim)
            )
        else:
            self.reduction = nn.Sequential(
                nn.Dropout(dropout_fc),
                nn.Linear(2 * d_hidden, feat_dim),
                # nn.ReLU(),
                nn.Linear(feat_dim, num_classes)
            )

    def forward(self, x):
        emb_ctx = self.encoder(x)  # emb_ctx[bsz, max_len, d_ctx*2]
        emb_aggregate = self.aggregation(emb_ctx)  # [bsz, 2*d_ctx]
        emb_reduction = self.reduction(emb_aggregate)  # [bsz, dim_feature]
        return emb_reduction


class LGMLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=1, lambda_=0.5, **kwargs):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.softmax = nn.Softmax()
        self.CE = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))

    def forward(self, feat, labels=None, device=None, class_emb=None, *args, **kwargs):

        batch_size = feat.size()[0]
        # means = self.means if class_emb is None else class_emb
        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        if labels is not None:
            labels_reshped = labels.view(labels.size()[0], -1)  # [bsz] -> [bsz, 1]
            ALPHA = torch.zeros(batch_size, self.num_classes).to(device).scatter_(1, labels_reshped, self.alpha)  # margin
            K = ALPHA + torch.ones([batch_size, self.num_classes]).to(device)
            logits_with_margin = torch.mul(neg_sqr_dist, K)

            means = self.means if class_emb is None else class_emb
            means_batch = torch.index_select(means, dim=0, index=labels)
            loss_margin = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
            self.loss = self.CE(logits_with_margin, labels) + loss_margin
        return neg_sqr_dist


class SEG(nn.Module):
    def __init__(self, num_classes, feat_dim, p_y, alpha=1, lambda_=0.5, **kwargs):
        super(SEG, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.p_y = p_y
        self.softmax = nn.Softmax()
        self.CE = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))

    def forward(self, feat, labels=None, device=None, class_emb=None, *args, **kwargs):

        batch_size = feat.size()[0]
        # means = self.means if class_emb is None else class_emb
        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        # with p_y
        ########################################
        p_y = self.p_y.expand_as(neg_sqr_dist).to(device)  # [bsz, n_c_seen]
        dist_exp = torch.exp(neg_sqr_dist)
        dist_exp_py = p_y.mul(dist_exp)
        dist_exp_sum = torch.sum(dist_exp_py, dim=1, keepdim=True)  # [bsz, n_c_seen] -> [bsz, 1]
        logits = dist_exp_py / dist_exp_sum  # [bsz, n_c, seen]
        ########################################

        if labels is not None:
            labels_reshped = labels.view(labels.size()[0], -1)  # [bsz] -> [bsz, 1]
            ALPHA = torch.zeros(batch_size, self.num_classes).to(device).scatter_(1, labels_reshped, self.alpha)  # margin
            K = ALPHA + torch.ones([batch_size, self.num_classes]).to(device)

            #######################################
            dist_margin = torch.mul(neg_sqr_dist, K)
            dist_margin_exp = torch.exp(dist_margin)
            dist_margin_exp_py = p_y.mul(dist_margin_exp)
            dist_exp_sum_margin = torch.sum(dist_margin_exp_py, dim=1, keepdim=True)
            likelihood = dist_margin_exp_py / dist_exp_sum_margin
            loss_ce = - likelihood.log().sum() / batch_size
            #######################################

            means = self.means if class_emb is None else class_emb
            means_batch = torch.index_select(means, dim=0, index=labels)
            loss_gen = (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)

            ########################################
            self.loss = loss_ce + self.lambda_ * loss_gen
        return logits


class LMCL(nn.Module):
    def __init__(self, num_classes, feat_dim, s=30, m=0.35, **kwargs):
        super(LMCL, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.weights)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, *args, **kwargs):
        assert feat.size(1) == self.feat_dim, 'embedding size wrong'
        logits = F.linear(F.normalize(feat), F.normalize(self.weights))
        if labels is not None:
            margin = torch.zeros_like(logits)
            index = labels.view(-1, 1).long()
            margin.scatter_(1, index, self.m)
            m_logits = self.s * (logits - margin)
            self.loss = self.CE(m_logits, labels)
        return logits


class LSoftmax(nn.Module):
    def __init__(self, num_classes, feat_dim, **kwargs):
        super(LSoftmax, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, *args, **kwargs):
        logits = self.fc(feat)
        if labels is not None:
            self.loss = self.CE(logits, labels)
        return logits


class MSP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MSP, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, *args, **kwargs):
        logits = feat
        if labels is not None:
            self.loss = self.CE(logits, labels)
        return logits

    def predict(self, feat, threshold=0.5):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: outlier_np: numpy array [n_test] in {-1, 1}
        prop = self.softmax(torch.from_numpy(feat))
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        outlier_np = np.zeros(confidence_np.shape)
        outlier_np[confidence_np < threshold] = -1
        outlier_np[confidence_np >= threshold] = 1
        return outlier_np

    def score_samples(self, feat):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: confidence_np: numpy array [n_test]
        prop = self.softmax(torch.from_numpy(feat))
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        return confidence_np


class DOC(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DOC, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, device=None, *args, **kwargs):
        logits = self.sigmoid(feat)
        # logits = feat
        if labels is not None:
            batch_size, n_c = logits.shape
            onehot = torch.zeros(batch_size, n_c).to(device).scatter_(1, labels.unsqueeze(1), 1)
            onehot_neg = torch.ones(batch_size, n_c).to(device).scatter_(1, labels.unsqueeze(1), 0)
            p = logits.mul(onehot) - logits.mul(onehot_neg) + onehot_neg + 0.0001
            p_log = p.log()
            self.loss = torch.sum(-p_log)
            if self.loss.item() != self.loss.item():
                # print('nan')
                pass
            # self.loss = self.CE(logits, labels)
        return logits

    def predict(self, feat, threshold=0.5):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: outlier_np: numpy array [n_test] in {-1, 1}
        prop = self.sigmoid(torch.from_numpy(feat))
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        outlier_np = np.zeros(confidence_np.shape)
        outlier_np[confidence_np < threshold] = -1
        outlier_np[confidence_np >= threshold] = 1
        return outlier_np

    def score_samples(self, feat):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: confidence_np: numpy array [n_test]
        prop = self.sigmoid(torch.from_numpy(feat))
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        return confidence_np
