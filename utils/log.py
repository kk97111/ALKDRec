import time
import pickle
import torch
class LOG(object):
    def __init__(self,args):
        self.args = args
        self.time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        self.log_path = './log/%s+%s+%s.txt'%(self.args.dataset,self.args.model,self.time)
        self.saved_path = './saved/%s+%s+%s.txt'%(self.args.dataset,self.args.model,self.time)
        exp_name = "%s+%s+%s"%(self.args.dataset,self.args.model,self.time)
        self.best_model_path = "./best_model/" + exp_name  # 模型保存路径名
        #创建log文件
        f = open(self.log_path, 'w')
        f.close()
        # 输出log文件
        Log ={'dataset:':args.dataset,'model':self.args.model}
        try:
            Log['hidden_factor'] = self.args.hidden_factor
        except:
            pass
        try:
            Log['hidden_factor_T'] = self.args.hidden_factor_T
        except:
            pass
        try:
            Log['hidden_factor_S'] = self.args.hidden_factor_S
        except:
            pass
        self.write_str(str(Log))
        
    def write_str(self,c):
        with open(self.log_path,'a') as f:
            f.write(c+'\n')    
#     def write_dict(self,c):
#         with open(self.log_path,'a') as f:
#             for key in c:
#                 f.write(c[key]+'\n')

def save_model(model, Path):
    torch.save(model.state_dict(), Path + '.pt')


def load_model(model, path):
    model.load_state_dict(torch.load(path + '.pt'))
    print('model loaded from %s' % path)
