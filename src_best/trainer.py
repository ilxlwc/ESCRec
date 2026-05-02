# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, args):

        self.args = args
        # self.cuda_condition = torch.cuda.is_available() and not args.no_cuda
        # self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model

        self.batch_size = args.batch_size
        self.sim = args.sim

        # if self.cuda_condition:
        #     self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.train_iteration(epoch, self.train_dataloader)

    def test(self, epoch, full_sort=False):
        return self.test_iteration(epoch, self.test_dataloader, full_sort=full_sort)

    def train_iteration(self, epoch, dataloader, full_sort=False):
        raise NotImplementedError

    def test_iteration(self, epoch, dataloader, full_sort=False):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, labels, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(labels, pred_list, k))
            ndcg.append(ndcg_k(labels, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        # self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label=label.view(1,-1)
        label=label.expand((2,label.shape[-1])).reshape(1,-1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask==0

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask==0]=float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        # labels = torch.zeros(N).to(positive_samples.device).long()
        labels = torch.zeros(N).to(positive_samples).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def GCL_loss(self, hidden1, hidden2, hidden_norm=True, temperature=1.0):
        # batch_size = hidden.shape[0] // 2
        # LARGE_NUM = 1e9
        # # inner dot or cosine
        # if hidden_norm:
        #     hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        # hidden_list = torch.split(hidden, batch_size, dim=0)
        # hidden1, hidden2 = hidden_list[0], hidden_list[1]

        batch_size = hidden1.shape[0]
        LARGE_NUM = 1e9
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size))
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = nn.CrossEntropyLoss()(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = nn.CrossEntropyLoss()(torch.cat([logits_ba, logits_bb], 1), labels)
        # loss = (loss_a + loss_b) * 0.5
        loss = (loss_a + loss_b) * 0.5
        return loss

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        return torch.matmul(z1, z2.t())

    def semi_loss(self, z1, z2, temperature=1.0):
        f = lambda x: torch.exp(x / temperature)
        refl_sim = f(self.sim(z1, z2))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    # Referee: Deep Graph Contrastive Representation Learning
    def DGC_loss(self, hidden1, hidden2, mean = True):

        l1 = self.semi_loss(hidden1, hidden2)
        l2 = self.semi_loss(hidden2, hidden1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self,coarse_intent_1, coarse_intent_2, target_item):
        sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :],
                                                       self.args.local_temperature, coarse_intent_1.shape[0], self.sim,
                                                       target_item[:, -1])
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss


class RecTrainer(Trainer):
    def __init__(self, model, train_dataloader, test_dataloader, args):
        super(RecTrainer, self).__init__(model, train_dataloader, test_dataloader, args)

    def train_iteration(self, epoch, dataloader, full_sort=True):
        # str_code = "train" if train else "test"
        # ------ model training -----#
        print("Performing Rec model Training:")
        self.model.train()
        rec_avg_loss = 0.0
        joint_avg_loss = 0.0
        con_sum_loss = 0.0

        global_sum_loss = 0.0
        local_sum_loss = 0.0

        print(f"rec dataset length: {len(dataloader)}")
        rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        print("")

        for i, (rec_batch) in rec_cf_data_iter:
            """             
            rec_batch shape: key_name x batch_size x feature_dim
            """
            # 0. batch_data will be sent into the device(GPU or CPU)
            # rec_batch = tuple(t.to(self.device) for t in rec_batch)

            # user_id, items_list, target_list, target_list_, label
            # origin_label: [0, 1, 2, 3, 4, 5, 6]
            # train_label = 5
            # items_list = [0, 1, 2, 3, 4]
            # target_list = [1, 2, 3, 4, 5]
            # target_list_ = [a, b, c, d, e(, 5)]
            user_id, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch

            # ---------- prediction task -----------------------#
            intent_output = self.model.predict(subsequence_1)

            print("intent_output\t{}".format(intent_output.shape))

            logits = self.predict_full(intent_output[:, -1, :])  # [Bx|I|]

            # print("logits: {},\t\ttarget_pos_1: {}".format(logits.shape, target_pos_1[:, -1].shape))
            rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

            # ---------- intent representation learning task ---------------#
            ori_local, aug_local, ori_global_sum, aug_global_sum = self.model(subsequence_1, subsequence_2)


            ##////////////////////////////////////////////////////////
            #去除对比损失
            # local_loss = self.cicl_loss(ori_local, aug_local, target_pos_1)
            global_loss = self.GCL_loss(ori_global_sum, aug_global_sum, hidden_norm=True, temperature=self.args.global_temperature)

            print("local_loss: {},\t\tglobal_loss: {}".format(local_loss, global_loss))

            # con_loss = self.args.lambda_0 * local_loss + self.args.beta_0 * global_loss
            con_loss = self.args.lambda_0 * local_loss + self.args.beta_0 * global_loss

            # ---------- multi-task learning --------------------#
            joint_loss = self.args.rec_weight * rec_loss + con_loss

            # print("local_loss: {}".format(local_loss))
            # print("global_loss: {}".format(global_loss))
            # print("joint_loss: {}".format(joint_loss))
            # print("con_loss: {}".format(con_loss))

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            rec_avg_loss += rec_loss.item()
            if type(con_loss) != float:
                con_sum_loss += con_loss.item()
            else:
                con_sum_loss += con_loss
            joint_avg_loss += joint_loss.item()

            global_sum_loss += global_loss.item()
            local_sum_loss += local_loss.item()

        post_fix = {
            "epoch": epoch,
            "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
            "icl_avg_loss": "{:.4f}".format(con_sum_loss / len(rec_cf_data_iter)),
            "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            "global_avg_loss": "{:.4f}".format(global_sum_loss / len(rec_cf_data_iter)),
            "local_avg_loss": "{:.4f}".format(local_sum_loss / len(rec_cf_data_iter)),
        }

        if (epoch + 1) % self.args.log_freq == 0:
            print(str(post_fix))

        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")

    def test_iteration(self, epoch, dataloader, full_sort=True):
        print("Performing Rec model testing:")

        rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        print("")
        self.model.eval()

        pred_list = None

        if full_sort:
            label_list = None
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # batch = tuple(t.to(self.device) for t in batch)

                # user_id, items_list, target_list, target_list_, label
                # origin_label: [0, 1, 2, 3, 4, 5, 6]
                user_ids, items_list, _, labels = batch
                recommend_output = self.model.predict(items_list)

                # recommend_output = self.model(items_list) # [BxLxH]
                recommend_output = recommend_output[:, -1, :] # [BxH]

                # recommendation results
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.test_matrix[batch_user_index].toarray() > 0] = 0

                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition T: O(n)  argsort O(nlogn)
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    label_list = labels.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    label_list = np.append(label_list, labels.cpu().data.numpy(), axis=0)
            return self.get_full_sort_score(epoch, label_list, pred_list)

        else:
            for i, batch in rec_data_iter:
                # batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, labels, sample_negs = batch
                recommend_output = self.model.finetune(input_ids)
                test_neg_items = torch.cat((labels, sample_negs), -1)
                recommend_output = recommend_output[:, -1, :]

                test_logits = self.predict_sample(recommend_output, test_neg_items)
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)






