# import torch
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader, RandomSampler
#
# from utils import get_user_seqs
# from datasets import DS, RecWithContrastiveLearningDataset
# import argparse
#
# args = argparse.ArgumentParser().parse_args()
# args.data_dir = "../data/"
# args.data_name = "Sports_and_Outdoors"
# args.max_seq_length = 50
#
# args.train_data_file = args.data_dir + args.data_name + "_aug_seq.txt"
# _, train_user_seq, _, _, _ = get_user_seqs(args.train_data_file)
# train_dataset = RecWithContrastiveLearningDataset(args, train_user_seq, data_type="train")
# train_sampler = RandomSampler(train_dataset)
# data = DataLoader(train_dataset, batch_size=10, sampler=train_sampler)
#
# # user_id, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch
# for i, (rec_batch) in enumerate(data):
#     user_id, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch
#     print(subsequence_1)
#     # print(' batch:{0} x:{1}  y: {2}'.format(i, x, y))
#
#     attention_mask = (subsequence_1 > 0).long()
#     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#     max_len = attention_mask.size(-1)
#     attn_shape = (1, max_len, max_len)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#     subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#     subsequent_mask = subsequent_mask.long()
#
#     extended_attention_mask = extended_attention_mask * subsequent_mask
#     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#
#     print(extended_attention_mask)
#
#     break


# import pickle
# from utils import get_user_seqs
# import random
#
# index = 50
#
# data_path = "../data/Sports_and_Outdoors_aug_seq_t.pkl"
# train_data_file = "../data/Sports_and_Outdoors_aug_seq.txt"
# with open(data_path, 'rb') as read_file:
#     data_dict = pickle.load(read_file)
# train_tag = data_dict["train"]
#
# # training data
# _, user_seq, _, _, _ = get_user_seqs(train_data_file)
# items = user_seq[index]
#
# temp = train_tag[items[-3]]
# print("//"*50)
# print(items)
# print("//"*50)
# print(items[-3])
# print("//"*50)
# print(temp)
#
# input_ids = items[:-3]
# target_pos = items[1:-2]
# flag = False
# for t_ in temp:
#     if t_[1:] == items[:-3]:
#         continue
#     else:
#         target_pos_ = t_[1:]
#         flag = True
# if not flag:
#     target_pos_ = random.choice(temp)[1:]
# print("//"*50)
# print("//"*50)
# print(target_pos_)
# print(target_pos)
# print(input_ids)
#
# pad_len = 10 - len(target_pos_)
# copied_input_ids = [0] * pad_len+target_pos_
# copied_input_ids = copied_input_ids[-10:]
# print(copied_input_ids)


import networkx as nx
from torch_geometric.data.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import NeighborSampler
import torch

graph = nx.Graph()
graph.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 2)])
# nx.draw_kamada_kawai(graph, with_labels=True)
data = from_networkx(graph)
print(data.edge_index)

# subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.neig_sample,
#                                            shuffle=False, num_workers=0, batch_size=items.shape[0])

items = torch.tensor([2])
print("items:{}".format(items.shape))
loader = NeighborSampler(edge_index=data.edge_index, node_idx=items, sizes=[2,2],  shuffle=False, num_workers=0, batch_size=items.shape[0])


g_adjs = []
print("//" * 50)
for (b_size, node_idx, adjs) in loader:
    print(b_size)
    print(node_idx)
    print(adjs)
print("##"*50)

