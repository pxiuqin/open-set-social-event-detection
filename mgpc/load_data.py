import os
import numpy as np
import torch
from scipy import sparse
import dgl
from torch.utils.data import Dataset

class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    # Used by remove_obsolete mode 1
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]
            self.matrix = self.matrix[:, indices_to_keep]


def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes

def generateMasks(length, data_split, i, validation_percent=0.2, test_percent=0.2, save_path=None):
    # verify total number of nodes
    print(length,data_split[i])
    assert length == data_split[i]
    if i==0:
        # randomly suffle the graph indices
        train_indices = torch.randperm(length)
        # get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(length * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices,save_path+'/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path+'/test_indices.pt')
        return train_indices, validation_indices, test_indices
    #If is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.arange(0, data_split[i], dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return test_indices


def getdata(embedding_save_path,data_split,i,args):
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)
    # load data
    data = SocialDataset(args.data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1]  # feature dimension

    g = dgl.DGLGraph(data.matrix,
                     readonly=True)
    num_isolated_nodes = graph_statistics(g, save_path_i)
    g.set_n_initializer(dgl.init.zero_initializer)
    # g.readonly(readonly_state=True)

    mask_path = save_path_i + '/masks'
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)

    # 这里是初始化训练时，要给定三个集合
    if i == 0:
        train_indices, validation_indices, test_indices = generateMasks(len(labels), data_split,i,
                                                                              args.validation_percent,
                                                                              args.test_percent,
                                                                              mask_path)
    else:
        test_indices = generateMasks(len(labels), data_split, i, args.validation_percent,
                                                                              args.test_percent,
                                                                              mask_path)
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    if args.use_cuda:
        g = g.to(device)
        features, labels = features.cuda(), labels.cuda()
        test_indices = test_indices.cuda()
        if i==0:
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()

    g.ndata['features'] = features
    g.ndata['labels'] = labels

    # 计算边的时间权重
    def compute_time_weight(edges):
        H = 100000
        M = 86400
        HM = 3600 * args.days
        # 计算源节点和目标节点特征的差值
        feature_diff = 1 / (1 + torch.abs(
            (edges.src['features'][:,-2]*H*M + edges.src['features'][:,-1]*M) 
            - (edges.dst['features'][:,-2]*H*M + edges.dst['features'][:,-1]*M)
                                 ) / HM)   # 转化成小时维度（一般一个事件时间跨度在1个小时左右）
        # 将差值作为时间权重
        return {'time_weight': feature_diff}

    if i==0:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices, None # ,data.matrix.toarray()
    else:
        g.apply_edges(compute_time_weight)   # 如果是非0 block，也就是微调阶段需要加上一个边的时间权重
        
        return save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices, None  # , data.matrix.toarray()

