import faiss
import time
error_flag = {'sig':0}
import math
import torch

def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()

def evaluate(model, test_data, k=20, args=None):
    #建立faiss检索库
    topN = k # 评价时选取topN
    gpu_indexs = [None]
    for i in range(1000):
        try:
            item_embs = model.output_items().cpu().detach().numpy()
            res = faiss.StandardGpuResources()  # 使用单个GPU
            flat_config = faiss.GpuIndexFlatConfig()
            #flat_config.device = 0#device

            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, model.item_emb.weight.size()[-1], flat_config)  # 建立GPU index用于Inner Product近邻搜索
            gpu_indexs[0].add(item_embs) # 给index添加向量数据
            if error_flag['sig'] == 0:
                break
            else:
                print("core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("error received", e)
        print("Faiss re-try", i)
        time.sleep(5)
        
    #
    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for _, ( targets, items, mask,__) in enumerate(test_data): # 一个batch的数据
        # 获取用户嵌入
        user_embs = model(to_tensor(items, 'cuda'), to_tensor(mask, 'cuda'),train=False)[0]
        user_embs = user_embs.cpu().detach().numpy()
        gpu_index = gpu_indexs[0]
        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2: # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, 100) # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                #remove item in festure;
                candidate_set = []
                for item_candidate in I[i]:
                    if item_candidate not in items[i]:
                        candidate_set.append(item_candidate)
                item_list = set(candidate_set[:topN])


                #item_list = set(I[i]) # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
                for no, iid in enumerate(item_list): # 对于每一个label物品
                    #print(iid_list)
                    if iid == iid_list: # 如果该label物品在近邻搜索的结果中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        total += len(targets) # total增加每个批次的用户数量

    recall = total_recall / total # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total # NDCG
    hitrate = total_hitrate * 1.0 / total # 命中率
    return {'recall': recall, 'ndcg': ndcg}#, 'hitrate': hitrate
