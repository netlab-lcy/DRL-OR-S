import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import Sequential, GCNConv, GATv2Conv, GATConv
from torch_geometric.data import Batch, Data

from a2c_ppo_acktr.distributions import Categorical, MultiCategorical, MultiTypeCategorical, AttentionDist, MultiTypeAttentionDist, MultiTypeAttentionDist2
from a2c_ppo_acktr.utils import init

class Policy(nn.Module):
    def __init__(self, node_state_dim,  node_num, type_num, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self._node_num = node_num
        self._type_num = type_num

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            self.base = GNNBase(node_state_dim, type_num, hidden_size=16) 
        else:
            self.base = base
        
        self.dist = MultiTypeAttentionDist(self.base.output_size, type_num)

    def forward(self, inputs):
        raise NotImplementedError

    '''
    @param:
        inputs: x:[batch, node_num, state_size]; edge_index: [2, edge_num]
        condition_state: [batch, condition_state_size]
        node_index: target node
        type_index: [batch]
        adj_mask: [num_node] 0-1 vector indicate whether a node is the candidate next hop of the target node
    @retval:
        value: shape[batch, 1]
        action: shape[batch, action_shape] # action_shape=1
        action_log_probs: shape[batch, 1]
    '''
    def act(self, inputs, condition_state, node_index, type_index, adj_mask, deterministic=False):
        value, actor_features = self.base(inputs, condition_state, node_index, type_index)
        dist = self.dist(actor_features, node_index, adj_mask, type_index)
        
        if deterministic:
            # argmax
            action = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            action = dist.sample().unsqueeze(-1)

        action_log_probs = dist.log_prob(action.squeeze(-1))

        return value, action, action_log_probs.unsqueeze(-1)

    '''
    @param:
        inputs: x:[batch, node_num, state_size]; edge_index: [2, edge_num]
        condition_state: [batch, condition_state_size], actually we don't need condition state for get_value function
        node_index: target node
        type_index: [batch]
    @retval:
        value: shape[batch, 1]
    '''
    def get_value(self, inputs, condition_state, node_index, type_index):
        value, _ = self.base(inputs, condition_state, node_index, type_index)
        return value

    '''
    @param:
        inputs: x:[batch, node_num, state_size]; edge_index: [2, edge_num]
        condition_state: [batch, condition_state_size]
        action: [batch, action_shape] # action_shape should equal 1
        node_index: int
        type_index: [batch], vector to indicate the flow type
        adj_mask: [num_node]
        in DRL-OR-S action_shape = 1
    @retval:
        value: shape[batch, 1]
        action_log_probs: shape[batch, 1]
        dist_entropy: shape: torch.tensor(x) 
    '''
    def evaluate_actions(self, inputs, condition_state, action, node_index, type_index, adj_mask):
        x, edge_index = inputs.x, inputs.edge_index
        value, actor_features = self.base(inputs, condition_state, node_index, type_index)
        
        dist = self.dist(actor_features, node_index, adj_mask, type_index)
        action_log_probs = dist.log_prob(action.squeeze(-1)) 
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs.unsqueeze(-1), dist_entropy


class GNNBase(nn.Module):
    def __init__(self, num_inputs, type_num, condition_state_size=1, hidden_size=64):
        super(GNNBase, self).__init__()
        self._hidden_size = hidden_size
        self._condition_state_size = condition_state_size
        self._num_inputs = num_inputs
        self._type_num = type_num

        self.critic = Sequential('x, edge_index',
            [(GCNConv(num_inputs + type_num, hidden_size),'x, edge_index -> x'),
            nn.LeakyReLU(),
            ]
        )
        self.actor_readout = nn.Sequential(
            nn.Linear(num_inputs + condition_state_size, hidden_size),
            nn.LeakyReLU(),
        )
        # not applied in TPDS version
        self.critic_readout = nn.Sequential(
            nn.Linear(num_inputs + type_num, hidden_size),
            nn.LeakyReLU(),
        )
        self.critic_linear = nn.Linear(hidden_size, 1)

    @property
    def output_size(self):
        return self._hidden_size

    '''
    @param:
        inputs: x: shape[batch, num_node, input_size]; edge_index [2, edge_num]
        condition_state: shape[batch, num_node, condition_state_size]
        node_index: target node
        type_index: [batch]
    @retval:
        value: shape[batch, 1]
        hidden_actor: shape[batch, num_node, hidden_size]
    '''
    def forward(self, inputs, condition_state, node_index, type_index):
        type_one_hot = F.one_hot(type_index, self._type_num)
        xs, edge_index = inputs.x, inputs.edge_index
        bs = xs.size(0)
        num_node = xs.size(1)
        
        concat_input_critic = torch.cat([type_one_hot.unsqueeze(1).expand(-1, num_node, -1), xs], -1)
        batch_critic = Batch.from_data_list([Data(x=x, edge_index=edge_index) for x in concat_input_critic])
        hidden_critic = self.critic(batch_critic.x, batch_critic.edge_index).view(bs, num_node, -1)
        
        concat_input_actor = torch.cat([condition_state, xs], -1)
        hidden_actor = concat_input_actor
        hidden_actor = self.actor_readout(hidden_actor)

        return self.critic_linear(hidden_critic[:,node_index]), hidden_actor
    
    
