import torch.nn as nn
import torch.nn.functional as F
import torch

torch.set_default_dtype(torch.float32)

class GateLayer(nn.Module):
    def __init__(self, emb_size, emb_dropout=0):
        super(GateLayer, self).__init__()
        self.gating_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(emb_size, emb_size)))
        self.gating_bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, emb_size)))

        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        return self.emb_dropout(torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias)))
    
    def add(self, x):
        return self.forward(x)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False, phase='pretrain', weight=0):
        super(GATLayer, self).__init__()

        # 更加phase来判断是否为finetune
        # if phase == 'finetune':
        #     self.emb_gate = GateLayer(in_dim)
        # else:
        #     self.emb_gate = None

        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.phase = phase
        self.weight = weight
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    # def message_func(self, edges):
    #     # message UDF for equation (3) & (4)
    #     return {'z': edges.src['z'], 'e': edges.data['e']}

    # def reduce_func(self, nodes):
    #     # reduce UDF for equation (3) & (4)
    #     # equation (3)
    #     alpha = F.softmax(nodes.mailbox['e'], dim=1)
    #     # equation (4)
    #     h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    #     return {'h': h}
    
    # def edge_attention(self, edges):
    #     # edge UDF for equation (2)
    #     z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    #     a = self.attn_fc(z2)
    #     # 获取边的时间权重
    #     time_weight = edges.data['time_weight']
    #     # 将时间权重与注意力分数相乘
    #     a = a * time_weight.unsqueeze(-1)
    #     return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        message = {'z': edges.src['z'], 'e': edges.data['e']}
        
        # 传递时间权重
        if self.phase == 'finetune':
            message['time_weight'] =  edges.data['time_weight'].unsqueeze(1)
            # return {'z': edges.src['z'], 'e': edges.data['e'], 'time_weight': edges.data['time_weight'].unsqueeze(1)}
        
        return message

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        
        # equation (4)
        # 判断是否为finetune，如果是要进行 Temporal Prompt
        if self.phase == 'finetune':
            time_weight = F.softmax(nodes.mailbox['time_weight'], dim=1)   # 获取时间权重
            h = torch.sum((alpha*(1-self.weight) + time_weight*self.weight) * nodes.mailbox['z'], dim=1)   # 使用时间权重加权邻居节点的特征
        else:
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
            
        return {'h': h}
    
    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['features']
        # h = self.emb_gate(h) if self.emb_gate else h
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]
        blocks[layer_id].dstdata['z'] = z_dst
        blocks[layer_id].apply_edges(self.edge_attention)
        # equation (3) & (4)
        blocks[layer_id].update_all(  # block_id – The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        # nf.layers[layer_id].data.pop('z')
        # nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']  # residual connection
        return blocks[layer_id].dstdata['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False, phase='pretrain', weight=0):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual, phase, weight))
        self.merge = merge

    def forward(self, blocks, layer_id):
        head_outs = [attn_head(blocks, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False, phase='pretrain', weight=0):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual, phase, weight)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual, phase, weight)  # 一个Head做输出
    
    def forward(self, blocks):
        h = self.layer1(blocks, 0)
        h = F.elu(h)
        # print(h.shape)
        blocks[1].srcdata['features'] = h  # 把第0层的输出来更新第1层的src features
        h = self.layer2(blocks, 1)
        h = F.normalize(h, p=2, dim=1)

        return h  # 这里输出第1层的 features  维度为 out_dim 默认64

