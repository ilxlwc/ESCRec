# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

# data_dir, data_name, num_node, max_seq_length, sample_num, max_step_size=3

def build_WITG_from_trainset(data_dir, data_name, num_node, max_seq_length, max_step_size=3, use_renorm=True, use_scale=False):
    i_file = data_dir + data_name + ".txt"
    with open(i_file, "r+") as fr:
        dataset = fr.readlines()

    relation = []
    adj = [dict() for _ in range(num_node)]

    for data in dataset:
        items = list(map(int, data.split(' ')))
        # 将训练标签和验证标签不纳入全局图
        u_i, item_seq = items[0], items[1:-2]
        # 截取数据集长度
        item_seq = item_seq[-max_seq_length:]

        for k in range(1, max_step_size+1):
            for j in range(len(item_seq) - k):
                relation.append([item_seq[j], item_seq[j + k], k])
                relation.append([item_seq[j + k], item_seq[j], k])

    for temp in relation:
        if temp[1] in adj[temp[0]].keys():
            adj[temp[0]][temp[1]] += 1 / temp[2]
        else:
            adj[temp[0]][temp[1]] = 1 / temp[2]

    adj_pyg = []
    weight_pyg = []

    for t in range(1, num_node):
        x = [v for v in sorted(adj[t].items(), reverse=True, key=lambda x: x[1])]
        adj_pyg += [[t, v[0]] for v in x]
        if use_scale:
            t_sum = 0
            for v in x:
                t_sum += v[1]
            weight_pyg += [v[1] / t_sum for v in x]
        else:
            weight_pyg += [v[1] for v in x]

    adj_np = np.array(adj_pyg)
    adj_np = adj_np.transpose()
    edge_np = np.array([adj_np[0, :], adj_np[1, :]])
    x = torch.arange(0, num_node).long().view(-1, 1)
    edge_attr = torch.from_numpy(np.array(weight_pyg)).view(-1, 1)
    edge_index = torch.from_numpy(edge_np).long()
    Graph_data = Data(x, edge_index, edge_attr=edge_attr)
    print(Graph_data)
    if use_renorm:
        row, col = Graph_data.edge_index[0], Graph_data.edge_index[1]
        row_deg = 1. / degree(row, num_node, Graph_data.edge_attr.dtype)
        col_deg = 1. / degree(col, num_node, Graph_data.edge_attr.dtype)
        deg = row_deg[row] + col_deg[col]
        new_att = edge_attr * deg.view(-1, 1)
        Graph_data.edge_attr = new_att

    torch.save(Graph_data, data_dir + data_name + 'witg.pt')

if __name__ == '__main__':
    data_dir = "../data/"
    data_name = "Beauty"
    num_node = 12101+1
    max_seq_length = 50
    build_WITG_from_trainset(data_dir, data_name, num_node, max_seq_length)