import gc
from itertools import combinations
import math
import time
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# 用于GraphPrompt框架用
def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = graph_sizes.size(0)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        graph_sizes_list = graph_sizes.view(-1).tolist()
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
        for i, l in enumerate(graph_sizes_list):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask
    
#use prompt to finish step 1
class graph_prompt_layer_mean(nn.Module):
    def __init__(self):
        super(graph_prompt_layer_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result
    
class graph_prompt_layer_weighted(nn.Module):
    def __init__(self,max_n_num):
        super(graph_prompt_layer_weighted, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class graph_prompt_layer_weighted_matrix(nn.Module):
    def __init__(self,max_n_num,input_dim):
        super(graph_prompt_layer_weighted_matrix, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(input_dim,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result


# THIS WILL NOT RETURN NEGATIVE VALUE
def distance2center2(input,center):
    n = input.size(0)
    m = input.size(1)
    k = center.size(0)
    input_expand = input.reshape(n, 1, m).expand(n, k, m)
    center_expand = center.expand(n, k, m)
    temp = input_expand - center_expand
    temp = temp * temp
    distance = torch.sum(temp, dim=2)
    return distance


def center_embedding(input,node_labels,label_num=0,debug=False):
    node_labels = node_labels.unsqueeze(1)
    device=input.device
    label_num = torch.max(node_labels) + 1
    mean = torch.ones(node_labels.size(0), node_labels.size(1)).to(device)
    index = node_labels  # torch.tensor(node_labels, dtype=int).to(device)
    # mean = torch.ones(label_num, 1).to(device)
    # index = torch.tensor(node_labels,dtype=int).to(device)
    # index = torch.unsqueeze(index, dim=0)

    if debug:
        print(node_labels)

    _mean = torch.zeros(label_num, 1, device=device).scatter_add_(dim=0, index=index, src=mean)
    preventnan = torch.ones(_mean.size(), device=device)*0.0000001
    _mean = _mean+preventnan
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    c = c / _mean
    return c

def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1-lambda0)/(1+np.exp(-K*(t-T/2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1-lambda0) * t/T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1-lambda0) * (1 - math.cos(math.pi * t/T))/2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t-R*T, R*T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError


# 用于计算输入向量矩阵中每对向量之间的平方欧氏距离
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def get_triplets_new(embeddings, labels, adjacency_matrix):
    embeddings = embeddings.cpu()
    distance_matrix = pdist(embeddings)  # 计算每对向量的欧几里得距离
    distance_matrix = distance_matrix.cpu()

    labels = labels.cpu().data.numpy()
    triplets = []
    for anchor_index, _ in enumerate(labels): 
        connected_indices = np.where(adjacency_matrix[anchor_index] == 1)[0]    # 有边连接
        connected_indices = connected_indices[(connected_indices != anchor_index) & (connected_indices < labels.shape[0])]  # 必须小于label的长度
        disconnected_indices = np.where(adjacency_matrix[anchor_index] == 0)[0]   # 没有边连接
        disconnected_indices = disconnected_indices[disconnected_indices < labels.shape[0]] # 必须小于label的长度

        for postive_index in connected_indices:
            negative_index = np.random.choice(disconnected_indices)
            triplets.append([anchor_index, postive_index, negative_index])


    if len(triplets) == 0:
        # 这里可能需要更合理的处理方式，例如随机选择一个负样本
        triplets.append([anchor_index, postive_index, disconnected_indices[0].item()])
        print('------------------------------------------------------------------')

    triplets = np.array(triplets)
    return torch.LongTensor(triplets)

def pre_train_loss(node_features, node_label, adjacency_matrix=None, temperature=1.0):
    """
    计算prompt损失函数。
    
    :param node_features: 节点特征表示的张量，形状为 (batch_size, num_features)
    :param node_label: 实例的真实类别标签的张量，形状为 (batch_size,)
    :param adjacency_matrix: 图的邻接矩阵
    :param temperature: 温度参数，用于控制softmax的平滑程度
    :return: 计算得到的prompt损失值
    """    

    c_embedding = center_embedding(node_features, node_label)
    distance = distance2center2(node_features, c_embedding)

    distance = 1/F.normalize(distance, dim=1)
    # distance /= temperature   # 应用温度参数  

    pred = F.log_softmax(distance, dim=1)
    _pred = torch.argmax(pred, dim=1, keepdim=True).squeeze()
    
    # 计算log softmax
    # log_softmax = F.log_softmax(pred, dim=1)
    
    # 选择正确类别的log softmax值
    correct_log_softmax = pred.gather(1, node_label.unsqueeze(1)).squeeze(1)
    
    # 计算损失
    pred_loss = -correct_log_softmax.mean()

    # 结合Triplets损失
    triplets = get_triplets_new(node_features, node_label, adjacency_matrix)
    anchor, positive, negative = node_features[triplets[:, 0]], node_features[triplets[:, 1]], node_features[triplets[:, 2]] 
    exp_pos = torch.exp(F.cosine_similarity(anchor,positive)/temperature)
    exp_neg = torch.exp(F.cosine_similarity(anchor,negative)/temperature)
    cosine_similarity_loss = -1 * torch.log(exp_pos / (exp_pos + exp_neg))
    triplet_loss = cosine_similarity_loss.mean()

    # 对两种loss求平均
    loss = pred_loss + triplet_loss
    
    return loss, _pred

def prompt_loss_with_center(node_features, node_label, c_embedding, adjacency_matrix=None, loss_percent=0.5, temperature=1.0):
    """
    计算prompt损失函数。
    
    :param node_features: 节点特征表示的张量，形状为 (batch_size, num_features)
    :param node_label: 实例的真实类别标签的张量，形状为 (batch_size,)
    :param adjacency_matrix: 图的邻接矩阵
    :param temperature: 温度参数，用于控制softmax的平滑程度
    :return: 计算得到的prompt损失值
    """    

    distance = distance2center2(node_features, c_embedding)
    distance = 1/F.normalize(distance, dim=1)
    # distance /= temperature   # 应用温度参数  

    pred = F.log_softmax(distance, dim=1)
    _pred = torch.argmax(pred, dim=1, keepdim=True).squeeze()
    
    # 计算log softmax
    # log_softmax = F.log_softmax(pred, dim=1)
    
    # 选择正确类别的log softmax值
    correct_log_softmax = pred.gather(1, node_label.unsqueeze(1)).squeeze(1)
    
    # 计算损失
    pred_loss = -correct_log_softmax.mean()

    # 结合Triplets损失
    triplets = get_triplets_new(node_features, node_label, adjacency_matrix)
    anchor, positive, negative = node_features[triplets[:, 0]], node_features[triplets[:, 1]], node_features[triplets[:, 2]] 
    exp_pos = torch.exp(F.cosine_similarity(anchor,positive)/temperature)
    exp_neg = torch.exp(F.cosine_similarity(anchor,negative)/temperature)
    cosine_similarity_loss = -1 * torch.log(exp_pos / (exp_pos + exp_neg))
    triplet_loss = cosine_similarity_loss.mean()

    # 对两种loss求平均
    loss = loss_percent * pred_loss + (1-loss_percent) * triplet_loss
    
    return loss, _pred


    def __init__(self, labelnum,device):
        super(label2onehot, self).__init__()
        self.labelnum=labelnum
        self.device=device
    def forward(self,input):
        labelnum = torch.tensor(self.labelnum).to(self.device)
        index=torch.tensor(input,dtype=int).to(self.device)
        output = torch.zeros(input.size(0), labelnum).to(self.device)
        src = torch.ones(input.size(0), labelnum).to(self.device)
        output = torch.scatter_add(output, dim=1, index=index, src=src)
        return output