# -*- coding: utf-8 -*-

import os
import argparse
import pickle
from utils import set_seed, EarlyStopping
from datasets import DS, RecWithContrastiveLearningDataset, get_user_seqs
from trainer import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from build_witg import build_WITG_from_trainset
from model import *


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../data/", type=str)
    parser.add_argument("--data_name", default="ml-1m", type=str)
    parser.add_argument("--max_seq_length", default=50, type=int) #50
    parser.add_argument('--sample_num', type=int, default=12) #构建图时邻域数量
    # parser.add_argument('--neig_sample', type=int, default=6) #图卷积时邻域数量
    parser.add_argument("--neig_sample", default=[8, 4], type=list, help='gnn sample') #图卷积时各级节点邻域数量
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=64)
    # parser.add_argument('--n_hop', type=int, default=2)

    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--seed", default=2024, type=int)

    parser.add_argument('--dropout_gcn', type=float, default=0.5, help='Dropout rate.')  # [0, 0.2, 0.4, 0.6, 0.8]
    parser.add_argument("--eps", type=float, default=0.1, help="global graph perturbation penalty")
    # parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')  # [0, 0.5]
    # parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--activate', type=str, default='relu')
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")

    # model args
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # parser.add_argument("--no_cuda", action="store_true")
    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    parser.add_argument("--sim", default='dot', type=str, help="the calculate ways of the similarity.")
    parser.add_argument("--lambda_0", type=float, default=1.0, help="weight of coarse-grain intent contrastive learning task")
    parser.add_argument("--beta_0", type=float, default=1.0, help="weight of fine-grain contrastive learning task")


    ## contrastive learning task args
    parser.add_argument("--local_temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied.")
    parser.add_argument("--global_temperature", default=0.5, type=float, help="softmax temperature (default:  1.0) - not studied.")
    parser.add_argument("--rec_weight", type=float, default=1, help="weight of contrastive learning task")
    parser.add_argument("--f_neg", action="store_true", help="delete the FNM component (both in cicl and ficl)")

    # robustness experiments
    parser.add_argument("--noise_ratio", default=0.0, type=float, help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument("--edge_drop", default=0.3, type=float,
                        help="percentage of negative interactions in a sequence - robustness analysis")

    args = parser.parse_args()
    set_seed(args.seed)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + ".txt"
    args.train_data_file = args.data_dir + args.data_name + "_aug_seq.txt"
    print(args.data_file)
    print(args.train_data_file)

    # construct supervisory signals via DS(·) operation
    # 数据增广操作，输出为[userid, itemid1, itemid2, itemid3,...]
    if not os.path.exists(args.train_data_file):
        DS(args.data_file, args.train_data_file, args.max_seq_length)

    # training data
    _, train_user_seq, _, _ = get_user_seqs(args.train_data_file)
    # valid and test data
    _, user_seq, max_item, test_rating_matrix = get_user_seqs(args.data_file)

    args.num_node = max_item + 1
    # args.mask_id = max_item + 1
    print("//" * 50)
    print("max_item {}".format(max_item))

    # if not os.path.exists(args.data_dir + args.data_name + '_adj_' + str(args.sample_num) + '.pkl'):
    #     generateGlobalGraph(args.data_dir, args.data_name, args.num_node, 50, args.sample_num, 3)
    # globalGraphAdj = pickle.load(open(args.data_dir + args.data_name + '_adj_' + str(args.sample_num) + '.pkl', 'rb'))
    # globalGraphWeight = pickle.load(open(args.data_dir + args.data_name + '_weight_' + str(args.sample_num) + '.pkl', 'rb'))
    #
    # globalGraphAdj, globalGraphWeight = handle_adj(globalGraphAdj, args.num_node, args.sample_num, globalGraphWeight)

    if not os.path.exists(args.data_dir + args.data_name + 'witg.pt'):
        build_WITG_from_trainset(args.data_dir, args.data_name, args.num_node, args.max_seq_length)
    global_graph = torch.load(args.data_dir + args.data_name + 'witg.pt')

    # training data
    train_dataset = RecWithContrastiveLearningDataset(args, train_user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    print("Init Model ...")
    model = Net(args, global_graph)
    print("Init RecTrainer ...")
    trainer = RecTrainer(model, train_dataloader, test_dataloader, args)
    print("")
    args.test_matrix = test_rating_matrix

    # print("//"*50)
    # print("**"*50)
    # print("starting trainer ...")
    # trainer.train(1)
    #
    # print("//"*50)
    # print("**"*50)
    # print("starting test ...")
    # args.test_matrix = test_rating_matrix
    # scores, result_info = trainer.test(0, full_sort=True)

    # save model args
    args.log_file = os.path.join(args.output_dir, f"log_{args.data_name}" + ".txt")
    # save model
    args.checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args.data_name}" + ".pt")

    print(f"Train ICSRec")
    early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)
        # evaluate on NDCG@20
        scores, _ = trainer.test(epoch, full_sort=True)
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0, full_sort=True)


if __name__ == "__main__":
    main()
