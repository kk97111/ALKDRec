import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from utils.Loss import BPRLoss

class FPMC(nn.Module):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=10, num_layers=3, dropout=0.1):
        super(FPMC, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = item_num
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.LI_emb = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # label embedding matrix
        self.item_emb = nn.Embedding(self.n_items, self.hidden_size)
        self.loss_func = BPRLoss()


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)


    def forward(self, item_list, mask, train=True):
        lengths = torch.reshape(mask.sum(dim=1),[-1,1])
        item_eb = self.LI_emb(item_list) # [b, s, h]
        user_eb = torch.sum(item_eb,dim=1) / lengths
        scores = self.calculate_score(user_eb)

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
        return loss
    def user_out_of_distribution_loss(self,item_list, mask):
        user_eb = self.forward(item_list, mask)[0]#[batch,d]
        user_eb = user_eb.unsqueeze(dim=1)#[batch,1,h]
        lengths = mask.sum(dim=1)
        item_eb = self.item_emb(item_list) # [b, s, h]
        similarity = torch.sigmoid(torch.sum(user_eb * item_eb,dim=-1))# [b, s]
        loss_user = torch.sum(similarity*mask,dim=-1) / lengths
        return loss_user
                
                