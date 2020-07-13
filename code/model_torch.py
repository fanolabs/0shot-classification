import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import model_encoder as encoder


class CapsAll(nn.Module):
    def __init__(self, config, config_spin, embedding=None):
        super(CapsAll, self).__init__()

        self.device = config['device']
        self.vocab_size = config['n_vocab']
        self.word_emb_size = config['d_emb']
        self.u = config_spin['d_lstm']

        self.encoder = encoder.LstmLayer(config_spin['n_layer'], self.u, self.word_emb_size,
                                         if_freeze=True, n_vocab=self.vocab_size, embedding=embedding)

        # self.encoder = encoder.TransformersLayer(config['key_pretrained'], self.u)

        self.drop = nn.Dropout(config_spin['dropout_emb'])

        # parameters for self-attention
        self.d = self.word_emb_size
        self.d_a = config_spin['d_a']
        self.r = config_spin['r']
        self.alpha = config_spin['alpha']

        # attention
        self.WS1 = nn.ModuleList()
        self.WS2 = nn.ModuleList()
        for Ar in range(self.r):
            self.WS1.append(nn.Linear(self.u * 2, self.d_a, bias=False).to(self.device))
            self.WS2.append(nn.Linear(self.d_a, self.u*2, bias=False).to(self.device))

        self.relu = nn.ReLU()

        self.s_cnum = config['n_seen_class']
        self.u_cnum = config['n_unseen_class']
        self.margin = config_spin['margin']
        self.keep_prob = config_spin['dropout_emb']
        self.num_routing = config_spin['n_routing']
        self.output_atoms = config_spin['output_atoms']

        # for capsule
        self.input_dim = self.r
        self.input_atoms = self.u * 2
        self.capsule_weights = nn.Parameter(torch.zeros((self.r, self.u * 2,
                                                         self.s_cnum * self.output_atoms)))
        self.init_weights()

    def forward(self, x):

        self.sentence_embedding = self.encoder_dim_att(x)

        # capsule
        output_dim = self.s_cnum
        dropout_emb = self.drop(self.sentence_embedding)
        input_tiled = torch.unsqueeze(dropout_emb, -1).repeat(1, 1, 1, output_dim * self.output_atoms)

        votes = torch.sum(input_tiled * self.capsule_weights, dim=2)
        votes_reshaped = torch.reshape(votes, [-1, self.input_dim, output_dim, self.output_atoms])
        input_shape = self.sentence_embedding.shape
        logit_shape = np.stack([input_shape[0], self.input_dim, output_dim])

        self.activation, self.weights_b, self.weights_c = self.routing(votes=votes_reshaped,
                                                                       logit_shape=logit_shape,
                                                                       num_dims=4)
        self.logits = self.get_logits()
        self.votes = votes_reshaped
        return

    def encoder_dim_att(self, x):

        outp = self.encoder(x).contiguous()
        size = outp.size()  # [BSZ, T, 2u]
        bsz = size[0]
        max_len = size[1]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*T, 2u]

        self.attention = []
        for Ar in range(self.r):
            self.attention.append(torch.zeros(bsz, self.u * 2, max_len))

        m = []
        for Ar in range(self.r):
            hbar = self.relu(self.WS1[Ar](self.drop(compressed_embeddings)))
            alphas = F.softmax(self.WS2[Ar](hbar).view(bsz, max_len, -1), dim=1)  # [bsz, len, hop]
            self.attention[Ar] = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, d_h*2, len]
            m.append(torch.sum(self.attention[Ar].mul(torch.transpose(outp, 1, 2)), dim=2))

        m = torch.stack(m)
        s_m = []
        for sample in range(bsz):
            s_m.append(m[:, sample, :])
        sentence_embedding = torch.stack(s_m, dim=0)
        self.sentence_reduction = torch.norm(sentence_embedding, dim=-1)
        # 0428
        # caps_shape = sentence_embedding.shape
        # caps_norm = torch.norm(sentence_embedding, dim=-1)
        # caps_norm_sum = torch.sum(caps_norm, dim=-1)
        # caps_norm_sum = caps_norm_sum.unsqueeze(-1).unsqueeze(-1).expand(caps_shape)
        # caps_norm_expand = caps_norm.unsqueeze(-1).expand(-1, -1, caps_shape[-1])
        # caps_agg = sentence_embedding.mul(caps_norm_expand)
        # caps_normalize = caps_agg / caps_norm_sum
        # self.sentence_reduction = caps_normalize.sum(1)
        return sentence_embedding

    def construct_unseen_CapW(self, sim):
        unseen_W = []
        BCapW = self.capsule_weights
        BCapW = torch.reshape(BCapW, [self.r, self.u * 2, self.s_cnum, self.output_atoms])
        BCapW = BCapW.permute([2, 0, 1, 3])  # 5 *r *2u *10
        for i in range(self.u_cnum):  # XIAOTONG
            sim_i = torch.reshape(sim[i, :], [-1, 1])
            newinput = BCapW.reshape(BCapW.shape[0], -1)
            newinput = sim_i.expand(-1, newinput.shape[1]).float() * newinput.float()
            inputshape = BCapW.shape
            input_tensor = newinput.reshape(inputshape[0], inputshape[1], inputshape[2], inputshape[3])
            unseen_W.append(torch.sum(input_tensor, dim=0))
        unseen_W = torch.stack(unseen_W, dim=0) # 2 *r *2u *10
        CapW = torch.cat((BCapW, unseen_W), 0)
        input_tensor = CapW.permute([1, 2, 0, 3])
        inputshape = input_tensor.shape
        self.CapW_all = input_tensor.reshape(inputshape[0], inputshape[1], inputshape[2] * inputshape[3])


    def predict(self, x):

        capsule_weights = nn.Parameter(self.CapW_all)
        output_dim = self.s_cnum + self.u_cnum
        self.sentence_embedding = self.encoder_dim_att(x)

        # capsule
        dropout_emb = self.drop(self.sentence_embedding)
        input_tiled = torch.unsqueeze(dropout_emb, -1).repeat(1, 1, 1, output_dim * self.output_atoms)
        votes = torch.sum(input_tiled * capsule_weights, dim=2)
        votes_reshaped = torch.reshape(votes, [-1, self.input_dim, output_dim, self.output_atoms])
        input_shape = self.sentence_embedding.shape
        logit_shape = np.stack([input_shape[0], self.input_dim, output_dim])
        self.activation, self.weights_b, self.weights_c = self.routing(votes=votes_reshaped,
                                                                       logit_shape=logit_shape,
                                                                       num_dims=4)
        self.logits = self.get_logits()
        self.votes = votes_reshaped
        unseen_logits = self.logits

        return unseen_logits

    def get_logits(self):
        logits = torch.norm(self.activation, dim=-1)
        # self.sentence_reduction = logits
        return logits

    def routing(self, votes, logit_shape, num_dims):
        votes_t_shape = [3, 0, 1, 2]
        for i in range(num_dims - 4):
            votes_t_shape += [i + 4]
        r_t_shape = [1, 2, 3, 0]
        for i in range(num_dims - 4):
            r_t_shape += [i + 4]

        votes_trans = votes.permute(votes_t_shape)
        logits = nn.Parameter(torch.zeros(logit_shape[0], logit_shape[1], logit_shape[2]))
        logits = logits.to(self.device)
        activations = []

        # Iterative routing.
        for iteration in range(self.num_routing):
            route = F.softmax(logits, dim=2)
            preactivate_unrolled = route * votes_trans
            preact_trans = preactivate_unrolled.permute(r_t_shape)
            # delete bias to fit for unseen classes
            preactivate = torch.sum(preact_trans, dim=1)
            activation = self._squash(preactivate)
            activations.append(activation)
            # distances: [batch, input_dim, output_dim]
            act_3d = activation.unsqueeze(1)
            tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
            tile_shape[1] = self.input_dim
            act_replicated = act_3d.repeat(tile_shape)
            distances = torch.sum(votes * act_replicated, dim=3)
            logits = logits + distances

        return activations[self.num_routing - 1], logits, route

    def _squash(self, input_tensor):
        norm = torch.norm(input_tensor, dim=2, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    def init_weights(self):

        for Ar in range(self.r):
            nn.init.xavier_uniform_(self.WS1[Ar].weight)
            nn.init.xavier_uniform_(self.WS2[Ar].weight)

        nn.init.xavier_uniform_(self.capsule_weights)

        for Ar in range(self.r):
            self.WS1[Ar].weight.requires_grad_(True)
            self.WS2[Ar].weight.requires_grad_(True)
        self.capsule_weights.requires_grad_(True)

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        """Penalizes deviations from margin for each logit.
        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.
        Args:
            labels: tensor, one hot encoding of ground truth.
            raw_logits: tensor, model predictions in range [0, 1]
            margin: scalar, the margin after subtracting 0.5 from raw_logits.
            downweight: scalar, the factor for negative cost.
        Returns:
            A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = labels * (logits < margin).float() * ((logits - margin) ** 2)
        negative_cost = (1 - labels) * (logits > -margin).float() * ((logits + margin) ** 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def loss(self, label):

        # Max Margin loss
        loss_val = self._margin_loss(label, self.logits)
        loss_val = torch.mean(loss_val)

        # Attention loss
        A1 = torch.stack(self.attention, 0)
        A = []
        for bsz in range(A1.size()[1]):
            A.append(A1[:, bsz, :, :])
        Attention = torch.stack(A, dim=0)
        Attention = torch.sum(Attention, dim=2)
        Attention = torch.div(Attention, self.u*2)

        self_atten_mul = torch.matmul(Attention, Attention.permute([0, 2, 1])).float()
        sample_num, att_matrix_size, _ = self_atten_mul.shape
        I = torch.from_numpy(np.identity(att_matrix_size)).float().to(self.device)

        self_atten_loss = (torch.norm(self_atten_mul-I).float()) ** 2

        if math.isnan(loss_val):
            print('capsule Loss NaN ')
        if math.isnan(self.alpha * torch.mean(self_atten_loss)):
            print('renomal loss NaN')
        return 1000 * loss_val + self.alpha * torch.mean(self_atten_loss)
