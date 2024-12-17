import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_, xavier_normal_
from utils.Loss import BPRLoss
from torch.nn.init import normal_
epsilon = 1e-4
class STAMP(nn.Module):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=10, num_layers=3, dropout=0.1):
        super(STAMP, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size
        self.n_items = item_num + 1 # 0 for None, so + 1
        self.n_layers = num_layers
        self.dropout = dropout
        # Embedding layer
        self.LI = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sf = nn.Softmax(dim=1) #nn.LogSoftmax(dim=1)
        
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = BPRLoss()


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    def count_alpha(self, context, aspect, output):
        r"""This is a function that count the attention weights
        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]
        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        output_3dim = output.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha
        
    def forward(self, item_list, mask, train=True):
        seq = item_list
        lengths = mask.sum(-1)
        batch_size = seq.size(-1)
        item_seq_emb = self.LI(seq) # [b, seq_len, emb]
#        last_inputs = self.gather_indexes(item_seq_emb, lengths - 1)
        lengths = torch.Tensor(lengths).to('cuda')
        item_last_click_index = lengths - 1
        item_last_click = torch.gather(seq, dim=1, index=item_last_click_index.unsqueeze(1).long()) # [b, 1]
        last_inputs = self.LI(item_last_click.squeeze())# [b, emb]
        org_memory = item_seq_emb # [b, seq_len, emb]
        ms = torch.div(torch.sum(org_memory, dim=1), lengths.unsqueeze(1).float())# [b, emb]
        alpha = self.count_alpha(org_memory, last_inputs, ms) * mask # [b, seq_len]
        vec = torch.matmul(alpha.unsqueeze(1), org_memory) # [b, 1, emb]
        user_eb = vec.squeeze(1) + ms # [b, emb]
        item_embs = self.item_emb(torch.arange(self.n_items).to('cuda'))
        scores = torch.matmul(user_eb, item_embs.permute(1, 0))
        item_scores = scores#self.sigmoid(scores)
        return user_eb, scores

    def calculate_score(self, user_eb):
        all_items = self.item_emb.weight
        scores = torch.matmul(user_eb, all_items.transpose(1, 0)) # [b, n]
        return scores
    def output_items(self):
        return self.item_emb.weight
    
    def loss(self,user_emb,target,negatives):
        embs_item_positive = self.item_emb(target)#[b,k]
        embs_item_negatives = self.item_emb(negatives)#[b,n_neg,k]
        pos_score = (user_emb * embs_item_positive).sum(dim=-1, keepdim=True)#[b,1]
        neg_score = (user_emb.unsqueeze(dim=1) * embs_item_positive).sum(dim=-1)#[b,n_neg] # all negative except the diag;
        loss = self.loss_func(pos_score,neg_score)
        # loss = self.loss_func(pos_score.view(-1),torch.ones_like(pos_score.view(-1))) + self.loss_func(neg_score.view(-1),torch.zeros_like(neg_score.view(-1)))
        return loss


    def user_out_of_distribution_loss(self,item_list, mask):
        user_eb = self.forward(item_list, mask)[0]#[batch,d]
        user_eb = user_eb.unsqueeze(dim=1)#[batch,1,h]
        lengths = mask.sum(dim=1)
        item_eb = self.item_emb(item_list) # [b, s, h]
        similarity = torch.sigmoid(torch.sum(user_eb * item_eb,dim=-1))# [b, s]
        loss_user = torch.sum(similarity*mask,dim=-1) / lengths
        return loss_user