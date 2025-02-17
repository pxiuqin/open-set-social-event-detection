import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Any

from graph_prompt import center_embedding

# 用于GPF框架用
def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    

class CenterEmbedding(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(CenterEmbedding, self).__init__()
        self.layer = nn.Linear(2* in_channels, in_channels)
        self.gpf = GPFplusAtt(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, labels):
        c_embedding = center_embedding(x, labels)
        c_embedding_prompt = center_embedding(self.gpf.add(x), labels)   # 这里按照论文可以直接把x换成c_embedding即可，然后不用这个Linear，直接出结果了

        center_outs = self.layer(torch.cat((c_embedding, c_embedding_prompt), dim=1))

        return center_outs

def embeddings(model, blocks):
    blocks[0].srcdata['features'] = model.add(blocks[0].srcdata['features'])
    return blocks