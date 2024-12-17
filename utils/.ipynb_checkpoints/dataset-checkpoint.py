import pandas as pd
import numpy as np
import toolz
import copy
from sklearn.utils import shuffle

import torch
from torch.utils.data import DataLoader
import random

np.random.seed(0)

    
class Dataset(object):
    def __init__(self,args):
        self.args = args
        self.dataset = args.dataset
        self.dir = './dataset/%s/'%self.dataset
        
        #读取meta数据for title, and id变换
        print('loading data: meta_data\n')
        meta_df = pd.read_csv('./datasets/%s/%s_title.csv'%(self.dataset,self.dataset))
        meta_df.itemID = meta_df.itemID.apply(str)
        meta_df = meta_df.drop_duplicates(subset=['itemID'])
        self.meta_processing(meta_df)
        
        #读取数据集
        print('loading data: interaction_data\n')
        df = pd.read_pickle('./datasets/%s/%s.pkl'%(self.dataset,self.dataset))
        
        #df = self.string2list(df,list_columns=['session','negatives'])
        #数据划分
        self.train, self.valid, self.test = self.data_split(df,ratio = [0.6,0.2,0.2])
    def string2list(self,df,list_columns):
        print('trans to list\n')
        for key in list_columns:
            list(map(lambda x: eval(x),df[key].values))
        return df
    
    def meta_processing(self,meta_df):
        self.item2title = {line[0]:line[1] for line in meta_df.values}
        self.title2item = {line[1]:line[0] for line in meta_df.values}
        self.item2id = {line[0]:i for i,line in enumerate(meta_df.values)}
        self.id2item = {i:line[0] for i,line in enumerate(meta_df.values)}
        self.title2id = {line[1]:i for i,line in enumerate(meta_df.values)}
        self.id2title = {i:line[1] for i,line in enumerate(meta_df.values)}        
        self.n_item = len(self.item2title)

    def data_split(self,df,ratio):
        print('split data\n')
        df = self.replace_in_df_with_mapping(df,self.item2id)
        df = df[df['session'].apply(lambda x: len(x) > 0)]
        size = len(df)
        n1,n2 =  int(size * ratio[0]), int(size *(ratio[0]+ratio[1]))
        df['sessionID'] = np.arange(len(df))

        
        train = df[:n1]
        valid = df[n1:n2]
        test = df[n2:]
        
        return train,valid,test
    
    def replace_in_df_with_mapping(self, df, mapping):
        """
        替换DataFrame中的字符串或字符串列表。

        :param df: 要处理的Pandas DataFrame
        :param mapping: 字符串到数字的映射字典
        :return: 更新后的DataFrame
        """
        df = copy.deepcopy(df)
        label_type =  type([mapping[key] for i,key in enumerate(mapping) if i==0][0])
        for col in df.columns:
            # 检查列中的数据类型
            if isinstance(df[col][0], list):
                # 列表类型，遍历列表替换
                df[col] = df[col].apply(lambda lst: [mapping[item] for item in lst if item in mapping])           
            else:
                df[col] = df[col].map(mapping)
        df = df.dropna()
        df['label'] = df['label'].astype(label_type)
        
        return df    
    def uniform_negative_sample(self,feature,n_neg):
        return np.random.randint(0,self.n_item,[len(feature),n_neg])
    def hard_negative_sample(self,session_id,negative_sampler,n_neg):
        sample_results = []
        for sess in session_id:
            sample_results.append(np.random.choice(negative_sampler[sess],5))
        sample_results = np.array(sample_results)
        sample_results = np.concatenate((sample_results,np.random.randint(0,self.n_item,[len(session_id),5])),axis=1)    
        return sample_results
class DataIterator(torch.utils.data.IterableDataset):

    def __init__(self, tvt_data,
                 batch_size=128,
                 seq_len=10,
                 train_flag=1,
                 time_span = 128
                ):
        print("Using time span", time_span)
        self.read(tvt_data) # 读取数据，获取用户列表和对应的按时间戳排序的物品序列，每个用户对应一个物品list;这个应该有
        # tvt_data is a dataframe with ['session','label','negatives'] where session is k-1 items that the user engaged in
        self.users = list(np.arange(len(tvt_data))) # 用户列表 session recommendation 没有users

        self.batch_size = batch_size # 用于训练
        self.eval_batch_size = batch_size # 用于验证、测试
        self.train_flag = train_flag # train_flag=1表示训练
        self.seq_len = seq_len # 历史物品序列的最大长度
        self.index = 0 # 验证和测试时选择用户的位置的标记,没啥用,都是从0开始
        print("total session:", len(self.users))

    def __iter__(self):
        return self
    
    # def next(self):
    #     return self.__next__()

    def read(self, tvt_data):
        feature = tvt_data['session'].values
        label = tvt_data['label'].values
        session_ids = tvt_data['sessionID'].values
        
        self.dict_graph = {i:f for i,f in enumerate(feature)}
        self.dict_label = {i:f for i,f in enumerate(label)}
        self.dict_id = {i:f for i,f in enumerate(session_ids)}
        


    def __next__(self):
        if self.train_flag == 1: # 训练
            user_id_list = random.sample(self.users, self.batch_size) # 随机抽取batch_size个user
        else: # 验证、测试，按顺序选取eval_batch_size个user，直到遍历完所有user
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_mask_list = []
        hist_item_list = []
        session_id_list = []
        for user_id in user_id_list:
            session_id_list.append(int(self.dict_id[user_id]))
            item_id_list.append(int(self.dict_label[user_id]))
            item_list = self.dict_graph[user_id] # 排序后的user的item序列for features;
            k = len(item_list)
            # k前的item序列为历史item序列
            if k >= self.seq_len: # 选取seq_len个物品
                hist_item_list.append(item_list[k-self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))

        # 返回用户列表（batch_size）、物品列表（label）（batch_size）、
        # 历史物品列表（batch_size，seq_len）、历史物品的mask列表（batch_size，seq_len）
        return item_id_list, hist_item_list, hist_mask_list,session_id_list
class DataIterator_few_shot(torch.utils.data.IterableDataset):

    def __init__(self, tvt_data,
                 batch_size=128,
                 seq_len=10,
                 train_flag=1,
                 time_span = 128
                ):
        print("Using time span", time_span)
        self.read(tvt_data) # 读取数据，获取用户列表和对应的按时间戳排序的物品序列，每个用户对应一个物品list;这个应该有
        # tvt_data is a dataframe with ['session','label','negatives'] where session is k-1 items that the user engaged in
        self.users = list(np.arange(len(tvt_data))) # 用户列表 session recommendation 没有users

        self.batch_size = batch_size # 用于训练
        self.eval_batch_size = batch_size # 用于验证、测试
        self.train_flag = train_flag # train_flag=1表示训练
        self.seq_len = seq_len # 历史物品序列的最大长度
        self.index = 0 # 验证和测试时选择用户的位置的标记,没啥用,都是从0开始
        print("total session:", len(self.users))

    def __iter__(self):
        return self
    
    # def next(self):
    #     return self.__next__()

    def read(self, tvt_data):
        feature = tvt_data['feature'].values
        label = tvt_data['target'].values
        session_ids = tvt_data['session_id'].values
        LLM_position = tvt_data['LLM_hit'].values
        teacher_pred = tvt_data['teacher_top_item'].values
        student_pred = tvt_data['student_top_item_'].values
        student_position_pred = tvt_data['student_position'].values
        teacher_position_pred = tvt_data['teacher_position'].values

        self.dict_graph = {i:f for i,f in enumerate(feature)}
        self.dict_label = {i:f for i,f in enumerate(label)}
        self.dict_id = {i:f for i,f in enumerate(session_ids)}
        self.dict_LLM_position = {i:f for i,f in enumerate(LLM_position)}
        self.dict_teacher_pred = {i:list(f) for i,f in enumerate(teacher_pred)}
        self.dict_student_pred = {i:list(f) for i,f in enumerate(student_pred)}
        self.dict_student_position_pred = {i:f for i,f in enumerate(student_position_pred)}
        self.dict_teacher_position_pred = {i:f for i,f in enumerate(teacher_position_pred)}
        


    def __next__(self):
        if self.train_flag == 1: # 训练
            user_id_list = random.sample(self.users, self.batch_size) # 随机抽取batch_size个user
        else: # 验证、测试，按顺序选取eval_batch_size个user，直到遍历完所有user
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_mask_list = []
        hist_item_list = []
        session_id_list = []
        LLM_position_list = []
        teacher_pred_list = []
        student_pred_list = []
        student_position_pred_list = []
        teacher_position_pred_list = []
        
        for user_id in user_id_list:
            session_id_list.append(int(self.dict_id[user_id]))
            # print(self.dict_student_pred[user_id])

            item_id_list.append(int(self.dict_label[user_id]))
            LLM_position_list.append(int(self.dict_LLM_position[user_id]))
            teacher_pred_list.append(self.dict_teacher_pred[user_id])
            student_pred_list.append(self.dict_student_pred[user_id])
            student_position_pred_list.append(self.dict_student_position_pred[user_id])
            teacher_position_pred_list.append(self.dict_teacher_position_pred[user_id])
            # print(student_pred_list)
            
            item_list = self.dict_graph[user_id] # 排序后的user的item序列for features;
            k = len(item_list)
            # k前的item序列为历史item序列
            if k >= self.seq_len: # 选取seq_len个物品
                hist_item_list.append(item_list[k-self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))

        # 返回用户列表（batch_size）、物品列表（label）（batch_size）、
        # 历史物品列表（batch_size，seq_len）、历史物品的mask列表（batch_size，seq_len）
        # print(student_pred_list)
        # for i,line in enumerate(student_pred_list):
        #     contains_tensor = any(isinstance(item, torch.Tensor) for item in line)
        #     print(contains_tensor)
        
        return item_id_list, hist_item_list, hist_mask_list,session_id_list,LLM_position_list,teacher_pred_list,student_pred_list,student_position_pred_list,teacher_position_pred_list
        
def get_DataLoader(source, batch_size=32, seq_len=10, train_flag=1, args=None):
    dataIterator = DataIterator(source, batch_size, seq_len, train_flag)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)
def get_DataLoader_few_shot(source, batch_size=32, seq_len=10, train_flag=1, args=None):
    dataIterator = DataIterator_few_shot(source, batch_size, seq_len, train_flag)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)