# -*- coding: utf-8 -*-

import random
import torch
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import copy


# Dynamic Segmentation operations
# 生成多条用户样本操作，每条样本代表用户的一个短期偏好行为
# 输出为[userid, itemid1, itemid2, itemid3,...]
def DS(i_file, o_file, max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :max_len: the max length of the sequence
    :return:
    """
    with open(i_file,"r+") as fr:
        data = fr.readlines()
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 2
    # save
    max_keep_len = max_len + 1
    for d_ in data:
        items = d_.split(' ')
        u_i, item = items[0], items[1:]
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 2
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - max_keep_len:
                j = start+3
                while j < len(item):
                    if start < 1 and j-start<max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start+max_save_len])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j+1])
                j += 1
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")

"""
生成标签及数据，每一个标签包含多条数据集
形式为：{"train":train_dic, "test":test_dic}
train_dic: taget_item:[[userid, item1, item2, item3, ...(taget_item)],[...]]
"""
class Generate_tag():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name+"_aug_seq"
        self.save_path = save_path
    def generate(self):
        data_f = self.path+"/"+self.data_name+".txt"
        train_dic={}
        test_dic={}
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(' ')
                tag_train = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train, [])
                train_dic[tag_train].append(train_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test, [])
                test_dic[tag_test].append(test_temp)

        total_dic = {"train":train_dic, "test":test_dic}
        print("Saving data to ", self.save_path)
        with open(self.save_path+"/"+self.data_name+"_t.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self,data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data=self.load_dict(data_path)
        return data[mode]

class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        # create target item sets
        self.sem_tag = Generate_tag(self.args.data_dir, self.args.data_name, self.args.data_dir)
        # train_tag: [[userid, item1, item2, item3, ...(taget_item)], [...]]
        self.train_tag = self.sem_tag.get_data(self.args.data_dir+"/"+self.args.data_name+"_aug_seq_t.pkl","train")
        self.true_user_id,_,_,_ = get_user_seqs(args.train_data_file)

    def _data_sample_rec_task(self, user_id, all_items_list, items_list, target_list, label):
        # make a deep copy to avoid original sequence be modified
        copied_items_list = copy.deepcopy(items_list)
        pad_len = self.max_len - len(copied_items_list)
        copied_items_list = [0] * pad_len + copied_items_list
        copied_items_list = copied_items_list[-self.max_len:]

        target_list_1, target_list_2 = None, None
        # train data
        if type(target_list) == tuple:
            pad_len_1 = self.max_len-len(target_list[1])
            target_list_1 = [0] * pad_len + target_list[0]
            target_list_2 = [0] * pad_len_1 + target_list[1]
            target_list_1 = target_list_1[-self.max_len:]
            target_list_2 = target_list_2[-self.max_len:]
            assert len(target_list_1) == self.max_len
            assert len(target_list_2) == self.max_len
        else:
            target_list = [0] * pad_len + target_list
            target_list = target_list[-self.max_len:]
            assert len(target_list) == self.max_len

        assert len(copied_items_list) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[user_id]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_items_list, dtype=torch.long),
                torch.tensor(target_list, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_list) == tuple:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_items_list, dtype=torch.long),
                    torch.tensor(target_list_1, dtype=torch.long),
                    torch.tensor(target_list_2, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                )
            else:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_items_list, dtype=torch.long),
                    torch.tensor(target_list, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                )
        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        # t_user_id = self.true_user_id[index]
        #用户行为序列
        all_items_list = self.user_seq[index]
        assert self.data_type in {"train", "test"}
        # [0, 1, 2, 3, 4, 5, 6]
        # train
        # items_list [0, 1, 2, 3, 4]
        # target_list [1, 2, 3, 4, 5]
        # target_list_ [a, b, c, d, e(, 5)]
        # test
        # input_ids [0, 1, 2, 3, 4, 5]
        # target [1, 2, 3, 4, 5, 6]
        # 采样一个同目标序列
        if self.data_type == "train":
            items_list = all_items_list[:-2]
            target_list = all_items_list[1:-1]
            # taget 为items[-2]的所有序列列表
            all_target_list = self.train_tag[all_items_list[-2]]
            target_list_ = random.choice(all_target_list)[1:]
            """ 此处需要改进 """
            label = [all_items_list[-2]]  # no use

            target_list = (target_list, target_list_)
            cur_rec_tensors = self._data_sample_rec_task(index, all_items_list, items_list, target_list, label)
            return cur_rec_tensors
        else:
            items_with_noise = self._add_noise_interactions(all_items_list)
            items_list = items_with_noise[:-1]
            target_list = items_with_noise[1:]
            label = [items_with_noise[-1]]

            cur_rec_tensors = self._data_sample_rec_task(index, items_with_noise, items_list, target_list, label)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


"""
max_seq_length:获取用户序列的最大长度
max_step_size：建立边的关系时，允许的最大图跨度
"""
def generateGlobalGraph(data_dir, data_name, num_node, max_seq_length, sample_num, max_step_size=3):
    # if data_name == 'Sports_and_Outdoors':
    #     num_node = 18357+1
    # elif data_name == "Toys_and_Games":
    #     num_node = 11924+1
    # elif data_name == "Beauty":
    #     num_node = 12101+1
    # elif data_name == "Beauty_tmp":
    #     num_node = 9370+1
    # elif data_name == "ml-1m":
    #     num_node = 3416+1
    # else:
    #     num_node = 3

    i_file = data_dir + data_name + ".txt"
    with open(i_file, "r+") as fr:
        dataset = fr.readlines()

    relation = []
    adj1 = [dict() for _ in range(num_node)]
    adj = [[] for _ in range(num_node)]

    for data in dataset:
        items = list(map(int, data.split(' ')))
        # 将训练标签和验证标签不纳入全局图
        u_i, item_seq = items[0], items[1:-2]
        # 截取数据集长度
        item_seq = item_seq[-max_seq_length:]

        for k in range(1, max_step_size+1):
            for j in range(len(item_seq) - k):
                relation.append([item_seq[j], item_seq[j + k]])
                relation.append([item_seq[j + k], item_seq[j]])

    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1

    weight = [[] for _ in range(num_node)]

    for t in range(num_node):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x]
        weight[t] = [v[1] for v in x]

    for i in range(num_node):
        adj[i] = adj[i][:sample_num]
        weight[i] = weight[i][:sample_num]

    pickle.dump(adj, open(data_dir + data_name + '_adj_' + str(sample_num) + '.pkl', 'wb'))
    pickle.dump(weight, open(data_dir + data_name + '_weight_' + str(sample_num) + '.pkl', 'wb'))

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    user_id=[]
    item_set = set()
    for line in lines:
        item_arr = line.strip().split(' ')
        user, items = item_arr[0], item_arr[1:]
        items = [int(item) for item in items]
        user_seq.append(items)
        user_id.append(int(user))
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)
    num_items = max_item+1
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_id, user_seq, max_item, test_rating_matrix

def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

if __name__ == "__main__":
    data_dir = "../data/"
    data_name = "Sports_and_Outdoors"
    sample_num = 12
    globalGraphFile = data_dir + data_name + '_adj_' + str(sample_num) + '.pkl'
    if not os.path.exists(globalGraphFile):
        generateGlobalGraph(data_dir, data_name, 50, sample_num, 3)
    adj = pickle.load(open(data_dir + data_name + '_adj_' + str(sample_num) + '.pkl', 'rb'))
    weight = pickle.load(open(data_dir + data_name + '_weight_' + str(sample_num) + '.pkl', 'rb'))
    # adj, num = handle_adj(adj, max_item_idx, sample_num, weight)





