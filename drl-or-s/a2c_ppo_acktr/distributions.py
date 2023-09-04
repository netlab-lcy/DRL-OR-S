import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import init


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


'''
the distribution layer which apply specific layer for each dst
'''
class MultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_node):
        super(MultiCategorical, self).__init__()
        
        linears = []
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        for i in range(num_node):
            linear = init_(nn.Linear(num_inputs, num_outputs))
            linears.append(linear)
        self.linears = nn.ModuleList(linears)

    '''
    @param:
        x: shape[(len, )batch, num_inputs]
        dst_state: shape[(len, )batch, num_node], num_node = num outputs
    @retval:
        a distribution: shape[(len, )batch, num_outputs]
    '''
    def forward(self, x, dst_state):
        xs = []
        for linear in self.linears:
            xs.append(linear(x))
        concat_x = torch.stack(xs, -2) 
        result = concat_x * dst_state.unsqueeze(-1)
        result = torch.sum(result, -2)
        
        return torch.distributions.Categorical(logits=result)


'''
the distribution layer which apply specific layer for each dst and each type 
layer num = m(type) + n(dst)
'''
class MultiTypeCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_node, num_type):
        super(MultiTypeCategorical, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        
        dst_type_linears = []
        for i in range(num_node * num_type):
            linear = init_(nn.Linear(num_inputs, num_outputs)) 
            dst_type_linears.append(linear)
        self.dst_type_linears = nn.ModuleList(dst_type_linears)
    
    '''
    @param:
        x: shape[(len,)batch, num_inputs]
        dst_state: shape[(len,)batch, num_node]
    @retval:
        a distribution: shape[(len, )batch, num_outputs]
    '''
    def forward(self, x, dst_state, type_state): 
        xs = []
        for linear in self.dst_type_linears:
            xs.append(linear(x))
        concat_x = torch.stack(xs, -2) 
        dst_type_state = torch.matmul(dst_state.unsqueeze(-1), type_state.unsqueeze(-2))
        state_shape = list(dst_type_state.size())
        if len(state_shape) == 4:
            dst_type_state = dst_type_state.view(state_shape[0], state_shape[1], -1)
        elif len(state_shape) == 3:
            dst_type_state = dst_type_state.view(state_shape[0], -1)
        else:
            raise NotImplementedError
        result = concat_x * dst_type_state.unsqueeze(-1)
        result = torch.sum(result, -2)
        return torch.distributions.Categorical(logits=result)
        

class AttentionDist(nn.Module):
    def __init__(self, num_inputs):
        super(AttentionDist, self).__init__()

        
        self.a = nn.Parameter(torch.zeros(size=(2*num_inputs, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    '''
        x: [batch, node_num, num_input]
        node_index: [2, N]
        adj_mask:[node_num] indicate the neighbor of target node
    '''
    def forward(self, x, node_index, adj_mask):
        node_num = x.size(1)
        concat_input = torch.cat([x[:, node_index].unsqueeze(1).expand(-1, node_num, -1), x], dim=-1)
        e = self.leakyrelu(torch.bmm(concat_input, self.a).squeeze(-1)) # batch * node_num
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1) * adj_mask.unsqueeze(0)
        return torch.distributions.Categorical(probs=e)
'''
Type-specific graph attention distribution
'''
class MultiTypeAttentionDist(nn.Module):
    def __init__(self, num_inputs, type_num):
        super(MultiTypeAttentionDist, self).__init__()
        self.type_num = type_num

        self.a = nn.Parameter(torch.zeros(size=(type_num, 2*num_inputs, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414) 

        self.a1 =  nn.Parameter(torch.zeros(size=(type_num, 2*num_inputs, 16)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(type_num, 16, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    '''
        x: [batch, node_num, num_input]
        node_index: int
        adj_mask:[batch] indicate the neighbor of target node
        type_index:[batch]
    '''
    def forward(self, x, node_index, adj_mask, type_index):
        type_one_hot = F.one_hot(type_index, self.type_num)
        attentions = []
        for i in range(self.type_num):
            node_num = x.size(1)
            concat_input = torch.cat([x[:, node_index].unsqueeze(1).expand(-1, node_num, -1), x], dim=-1)
            #e = self.leakyrelu(torch.matmul(concat_input, self.a[i].unsqueeze(0)).squeeze(-1)) # batch * node_num
            # for testing
            e = self.leakyrelu(torch.matmul(concat_input, self.a1[i].unsqueeze(0))) # batch * node_num * 64
            e = self.leakyrelu(torch.matmul(e, self.a2[i].unsqueeze(0)).squeeze(-1)) # batch * node_num

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj_mask > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1) * adj_mask.unsqueeze(0)
            attentions.append(attention)
        concat_x = torch.stack(attentions, -2) 
        result = concat_x * type_one_hot.unsqueeze(-1)
        result = torch.sum(result, -2)
        return torch.distributions.Categorical(probs=result)

'''
Type-specific graph attention distribution
Each node attend with the raw input of its neighbors
'''
class MultiTypeAttentionDist2(nn.Module):
    def __init__(self, num_inputs, type_num, raw_input_dim):
        super(MultiTypeAttentionDist2, self).__init__()
        self.type_num = type_num

        self.a = nn.Parameter(torch.zeros(size=(type_num, num_inputs + raw_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414) 
        
        self.leakyrelu = nn.LeakyReLU()

    '''
        x: [batch, node_num, num_input]
        raw_x: [batch, node_num, raw_input_dim]
        node_index: int
        adj_mask:[batch] indicate the neighbor of target node
        type_index:[batch]
    '''
    def forward(self, x, raw_x, node_index, adj_mask, type_index):
        type_one_hot = F.one_hot(type_index, self.type_num)
        attentions = []
        for i in range(self.type_num):
            node_num = x.size(1)
            concat_input = torch.cat([x[:, node_index].unsqueeze(1).expand(-1, node_num, -1), raw_x], dim=-1)
            e = self.leakyrelu(torch.matmul(concat_input, self.a[i].unsqueeze(0)).squeeze(-1)) # batch * node_num
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj_mask > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1) * adj_mask.unsqueeze(0)
            attentions.append(attention)
        
        concat_x = torch.stack(attentions, -2) 
        result = concat_x * type_one_hot.unsqueeze(-1)
        result = torch.sum(result, -2)
        return torch.distributions.Categorical(probs=result)