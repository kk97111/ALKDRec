import torch
import torch.nn as nn
import torch.nn.functional as F

# def softmax_neg(logit):
#     hm = 1.0 - torch.eye(*logit.size()).to('cuda')
#     logit = logit * hm
#     e_x = torch.exp(logit - logit.max(dim=1, keepdim=True).values) * hm
#     return e_x / e_x.sum(dim=1, keepdim=True)

# class LossFunction(nn.Module):
#     def __init__(self, loss_type='TOP1', bpr=0, use_cuda=False):
#         """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
#         super(LossFunction, self).__init__()
#         self.loss_type = loss_type
#         self.use_cuda = use_cuda
#         if loss_type == 'TOP1':
#             self._loss_fn = TOP1Loss()
#         elif loss_type == 'BPR':
#             self._loss_fn = BPRLoss()
#         elif loss_type == 'TOP1-max':
#             self._loss_fn = TOP1_max()
#         elif loss_type == 'BPR-max':
#             self._loss_fn = BPR_max(bpr)
#         else:
#             raise NotImplementedError

#     def forward(self, logit):
#         return self._loss_fn(logit)
# class TOP1Loss(nn.Module):
#     def __init__(self):
#         super(TOP1Loss, self).__init__()
#     def forward(self, logit):
#         """
#         Args:
#             logit (BxB): Variable that stores the logits for the items in the mini-batch
#                          The first dimension corresponds to the batches, and the second
#                          dimension corresponds to sampled number of items to evaluate
#         """
#         diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
#         loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
#         return loss

# class BPRLoss(nn.Module):
#     def __init__(self):
#         super(BPRLoss, self).__init__()

#     def forward(self, logit):
#         """
#         Args:
#             logit (BxB): Variable that stores the logits for the items in the mini-batch
#                          The first dimension corresponds to the batches, and the second
#                          dimension corresponds to sampled number of items to evaluate
#         """
#         # differences between the item scores
#         diff = logit.diag().view(-1, 1).expand_as(logit) - logit
#         # final loss
#         loss = -torch.mean(F.logsigmoid(diff))
#         return loss

# class BPR_max(nn.Module):
#     def __init__(self, bpr):
#         super(BPR_max, self).__init__()
#         self.bpr = bpr
#     def forward(self, logit):
#         logit_softmax = softmax_neg(logit) # F.softmax(logit, dim=1)
#         diff = logit.diag().view(-1, 1).expand_as(logit) - logit
#         loss = torch.mean(
#             -torch.log(torch.sum(torch.sigmoid(diff) * logit_softmax, dim=1) + 1e-24) + self.bpr * torch.sum((logit**2)*logit_softmax, axis=1))
#         return loss

# class TOP1_max(nn.Module):
#     def __init__(self):
#         super(TOP1_max, self).__init__()

#     def forward(self, logit):
#         logit_softmax = softmax_neg(logit) # F.softmax(logit, dim=1)
#         diff = logit -logit.diag().view(-1, 1).expand_as(logit)
#         y = logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2))
#         loss = torch.mean(torch.sum(y, dim=1))
#         return loss
class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class MFLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(MFLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = ((pos_score-torch.ones_like(pos_score)) ** 2).mean() + ((neg_score-torch.zeros_like(neg_score))** 2).mean()
        return loss
    
    
    