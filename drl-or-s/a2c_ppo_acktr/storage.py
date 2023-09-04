import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, action_space, node_state_size, condition_state_size, num_node):
        self.obs = torch.zeros(num_steps + 1, num_node, node_state_size) 
        self.condition_states = torch.zeros(num_steps, num_node, condition_state_size) 
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        self.masks = torch.zeros(num_steps, 1) # mask whether a node is on the path
        self.type_indexs = torch.zeros(num_steps, 1).to(torch.int64)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1 # the sampled action i.e. action index
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.num_steps = num_steps
        self.step = 0

    def set_graph(self, edge_index, adj_mask, node_index):
        self.edge_index = edge_index
        self.adj_mask = adj_mask
        self.node_index = node_index 

    def to(self, device):
        self.obs = self.obs.to(device)
        
        self.condition_states = self.condition_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.masks = self.masks.to(device)
        self.type_indexs = self.type_indexs.to(device)
        self.actions = self.actions.to(device)
        
        self.edge_index = self.edge_index.to(device)
        self.adj_mask = self.adj_mask.to(device)

    def insert(self, obs, condition_state, actions, action_log_probs, value_preds, rewards, masks, type_index):
        self.obs[self.step + 1].copy_(obs)
        self.condition_states[self.step].copy_(condition_state)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step].copy_(masks)
        self.type_indexs[self.step].copy_(type_index)
        

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.type_indexs[0].copy_(self.type_indexs[-1])

    '''
    computing self.returns
    self.returns is the target values for the states, 
    use_gae is highly recomdanded
    gae refer to: https://zhuanlan.zhihu.com/p/45107835
    '''
    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1]  - self.value_preds[step]
                gae = delta + gamma * gae_lambda * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma + self.rewards[step]

    '''
    @param:
        advantages: [num_step, 1]
        num_mini_batch: int
    @iterator retval:
        shape: shape of all the return value is [num_mini_batch, -1]
    '''
    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps = self.rewards.size()[0]
        batch_size = num_steps
        assert batch_size >= num_mini_batch, ("number of steps {}"
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[1:])[indices]
            condition_states_batch = self.condition_states.view(-1, *self.condition_states.size()[1:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ_batch = advantages.view(-1, 1)[indices]
            masks_batch = self.masks.view(-1, 1)[indices]
            type_indexs_batch = self.type_indexs.view(-1, 1)[indices]

            yield obs_batch, condition_states_batch, actions_batch, \
                value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ_batch, masks_batch, type_indexs_batch

