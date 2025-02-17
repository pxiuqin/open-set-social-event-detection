import joblib
from train import initial_train, prompt_train
import argparse
from metric import AccuracyMetric, MacroF1Metric
import torch
import os
import json
import numpy as np
from model_dynamic import GAT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('--prompt_epochs', default=5, type=int,
                        help="Number of prompt tuning epochs.")
    parser.add_argument('--n_epochs', default=15, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    
    parser.add_argument('--topk', default=50, type=int,
                        help="topK.")
    parser.add_argument('--days', default=1.0, type=float,
                        help="time interval.")
    parser.add_argument('--time_percent', default=0.6, type=float,
                        help="time percent.")
    parser.add_argument('--old_label_rate', default=0.4, type=float,
                        help="old event label rate.")
    parser.add_argument('--loss_percent', default=0.5, type=float,
                        help="loss percent.")
    
    parser.add_argument('--patience', default=10, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=2100, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--word_embedding_dim', type=int, default=302)   # French 514   English 386
    parser.add_argument('--hidden_dim', default=16, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=16, type=int,
                        help="Output dimension of tweet representations")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--gpf_p_num', default=8, type=int,
                        help="GPF attr num")
    parser.add_argument('--validation_percent', default=0.1, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--test_percent', default=0.2, type=float,
                        help="Percentage of test nodes(tweets)")
    parser.add_argument('--metrics', type=str, default='nmi')
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--gpuid', type=int, default=1)
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--is_incremental', default=True, action='store_true')
    parser.add_argument('--data_path', default='./data/0413_ALL_English',   # 0413_ALL_English  0413_ALL_French
                        type=str, help="Path of features, labels and edges")

    args = parser.parse_args()

    print("Using CUDA:", args.use_cuda)
    if args.use_cuda:
        torch.cuda.set_device(args.gpuid)

    embedding_save_path = args.data_path + '/embeddings_mgpc'
    if not os.path.exists(embedding_save_path):
        os.mkdir(embedding_save_path)
    print("embedding_save_path: ", embedding_save_path)
    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    data_split = np.load(args.data_path + '/data_split.npy')

    # Metrics
    metrics = [AccuracyMetric(),MacroF1Metric()]
    
    if args.is_incremental:
        model = GAT(args.word_embedding_dim, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual, 'finetune', args.time_percent)
        best_model_path = f"{embedding_save_path}/block_0/models/best.pt"   # 基于initial_train的最好模型开始进行下面的微调
        label_center = joblib.load(f"{embedding_save_path}/block_0/models/label_center.dump")

        # 加载预训练的模型参数
        state_dict = torch.load(best_model_path)
        model.load_state_dict(state_dict, strict=False)
        if args.use_cuda:
            model.cuda()

        old_label_rate = args.old_label_rate   # 初始化旧类占比

        for i in range(1, data_split.shape[0]):
            print("incremental setting")
            print("enter i ",str(i))

            _, label_center, old_label_rate = prompt_train(i, data_split, metrics, embedding_save_path, model, label_center, args, old_label_rate)
    else:
        # pre-training
       model = initial_train(0, args, data_split, metrics, embedding_save_path)