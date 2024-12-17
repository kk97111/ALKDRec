import torch
from tqdm import tqdm
from utils.dataset import Dataset,DataIterator,get_DataLoader

def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()


def calculate_session_embs(teacher_model,train,args):
    train_data_for_embedding =  get_DataLoader(train,args.batch_size, seq_len=10,train_flag=0)
    teacher_session_embs = torch.zeros([len(train),args.teacher_dims])
    for iter, (targets, items, mask, session_ids) in tqdm(enumerate(train_data_for_embedding)):
        targets_cuda = to_tensor(targets,'cuda')
        items_cuda = to_tensor(items,'cuda')
        mask_cuda = to_tensor(mask,'cuda')
        user_eb, scores = teacher_model(items_cuda,mask_cuda)
        teacher_session_embs[session_ids] = user_eb.cpu()
    return teacher_session_embs