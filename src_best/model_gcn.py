# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.loader import NeighborSampler


class GlobalGNN(nn.Module):
    def __init__(self, hidden_size, neig_sample, dropout_gcn):
        super(GlobalGNN, self).__init__()
        self.hidden_size = hidden_size
        in_channels = hidden_channels = self.hidden_size
        self.num_hop = len(neig_sample)
        self.dropout = nn.Dropout(dropout_gcn)
        self.gcn = GCNConv(self.hidden_size, self.hidden_size)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for i in range(self.num_hop - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))

        self.eps = 0.2

    def forward(self, em_nodes, adjs, attr, perturbed=False):
        # nodes: tensor([2, 3, 1, 4, 0])
        # em_nodes : nodes embedding
        # adjs: [EdgeIndex(edge_index=tensor([[2, 3, 2, 3, 1, 4], [0, 0, 1, 1, 2, 2]]), e_id=tensor([ 2, 11,  3, 10,  7,  0]), size=(5, 3)),
        #        EdgeIndex(edge_index=tensor([[1, 2], [0, 0]]), e_id=tensor([8, 2]), size=(3, 1))]

        # print("em_nodes{}".format(em_nodes.shape))

        output = []
        x_all = em_nodes

        if self.num_hop > 1:
            for i, (edge_index, e_id, s_size) in enumerate(adjs):
                # print("//"*50)
                # e_id 是在原图中的边id
                # s_size 表示本次batch中包含的采样后节点数（source+target）以及采样的节点数（source)
                weight = attr[e_id].view(-1).type(torch.float)

                # em_nodes = x_all
                if len(list(x_all.shape)) < 2:
                    x_all = x_all.unsqueeze(0)

                perturbed = True
                if perturbed:
                    random_noise = torch.rand_like(x_all)
                    x_all = x_all + torch.sign(x_all) * F.normalize(random_noise, dim=-1) * self.eps

                x_all = self.gcn(x_all, edge_index, weight)
                # sage
                x_target = x_all[:s_size[1]]  # Target nodes are always placed first.
                x_all = self.convs[i]((x_all, x_target), edge_index)
                if i != self.num_hop - 1:
                    x_all = F.relu(x_all)
                    x_all = self.dropout(x_all)

                # print("x_all{}".format(x_all.shape))
        else:
            # 只有1-hop的情況
            edge_index, e_id, s_size = adjs.edge_index, adjs.e_id, adjs.size
            # em_nodes = x_all
            x_all = self.dropout(x_all)
            weight = attr[e_id].view(-1).type(torch.float)
            if len(list(x_all.shape)) < 2:
                x_all = x_all.unsqueeze(0)
            x_all = self.gcn(x_all, edge_index, weight)
            x_target = x_all[:s_size[1]]  # Target nodes are always placed first.
            x_all = self.convs[-1]((x_all, x_target), edge_index)

        output.append(x_all)
        return torch.cat(output, 0)


class Net(nn.Module):
    def __init__(self, args, global_graph):
        super(Net, self).__init__()
        self.args = args

        self.global_graph = global_graph
        self.global_gnn = GlobalGNN(args.hidden_size, args.neig_sample, args.dropout_gcn)

        # Item representation & Position representation
        self.item_embeddings = nn.Embedding(args.num_node, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.item_encoder = Encoder(args)
        self.layerNorm = LayerNorm(args.hidden_size, eps=1e-12)

        self.linear_layer = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.apply(self.init_weights)

    def forward(self, ori_item_list, aug_item_list):
        ori_mask_item = (ori_item_list > 0).long()
        aug_mask_item = (aug_item_list > 0).long()

        ori_local = self.local_graph(ori_item_list, ori_mask_item)
        aug_local = self.local_graph(aug_item_list, aug_mask_item)

        ori_global = self.gnn_encode(ori_item_list.flatten())
        aug_global = self.gnn_encode(ori_item_list.flatten())
        ori_global = ori_global.view(-1, self.args.max_seq_length, self.args.hidden_size)
        aug_global = aug_global.view(-1, self.args.max_seq_length, self.args.hidden_size)
        ori_global_sum = torch.sum(ori_global * ori_mask_item.unsqueeze(-1), dim=1) / torch.sum(ori_mask_item.unsqueeze(-1).float(), dim=1)
        aug_global_sum = torch.sum(ori_global * ori_mask_item.unsqueeze(-1), dim=1) / torch.sum(ori_mask_item.unsqueeze(-1).float(), dim=1)

        global_info_hidden = torch.cat([ori_global_sum, aug_global_sum], 0)

        # ori_global = self.linear_layer(ori_global)
        # aug_global = self.linear_layer(ori_local)
        # print(' ori_global shape:{}'.format(ori_global.shape))
        # return shape : [batch_size, max_seq_length, hidden_size]
        return ori_local, aug_local, ori_global, aug_global, global_info_hidden
        # return ori_local, aug_local

    def local_graph(self, item_list, mask_item):
        ###########################################################
        # local attention
        extended_attention_mask = mask_item.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = mask_item.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        # if self.args.cuda_condition:
        #     subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        sequence_emb = self.add_position_embedding(item_list)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        l_local = item_encoded_layers[-1]
        return l_local


    def gnn_encode(self, items):
        # NeighborSampler用法参见 https://zhuanlan.zhihu.com/p/688064184
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.neig_sample,
                                           shuffle=False, num_workers=0, batch_size=items.shape[0])

        # subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.neig_sample,
        #                                    shuffle=True, num_workers=0, batch_size=items.shape[0])
        # print("gnn_encode {}".format(subgraph_loaders.shape))
        g_adjs = []
        em_nodes = []
        # only one layer
        for (b_size, node_idx, adjs) in subgraph_loaders:
            g_adjs = adjs
            em_nodes = self.item_embeddings(node_idx).squeeze()

        attr = self.global_graph.edge_attr
        g_hidden = self.global_gnn(em_nodes, g_adjs, attr)
        return g_hidden

    def predict(self, item_list):
        mask_item = (item_list > 0).long()
        l_local = self.local_graph(item_list, mask_item)
        return l_local

    def sample(self, v_target, n_sample):
        neighbor = self.adj_all[v_target.view(-1)]
        index = np.arange(neighbor.shape[1])
        np.random.shuffle(index)
        index = index[:n_sample]
        return self.adj_all[v_target.view(-1)][:, index], self.weight_all[v_target.view(-1)][:, index]
        # return self.adj_all[v_target.view(-1)], self.weight_all[v_target.view(-1)]


    # Positional Embedding
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.layerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


