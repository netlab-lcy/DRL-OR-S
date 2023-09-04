import copy
import glob
import os
import time
from collections import deque
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy, GNNBase
from a2c_ppo_acktr.storage import RolloutStorage

from net_env.simenv import NetEnv
from torch_geometric.data import Data


def main():
    # init 
    args = get_args()

    # for reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "/eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    ckpt_step = args.ckpt_steps # model save every ckpt_step
    

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # set up environment
    envs = NetEnv(args) 
    num_agent, num_node, observation_spaces, action_spaces, num_type, node_state_dim, agent_to_node, edge_indexs, adj_masks = envs.setup(args.env_name) # for GNN version, we don't need action space and observation_spaces for the model's training
    request, obses = envs.reset()

    # open log file
    log_dist_files = []
    log_demand_files = []
    log_delay_files = []
    log_throughput_files = []
    log_loss_files = []
    for i in range(num_type):
        log_dist_file = open("%s/dist_type%d.log" % (log_dir, i), "w", 1)
        log_dist_files.append(log_dist_file)
        log_demand_file = open("%s/demand_type%d.log" % (log_dir, i), "w", 1)
        log_demand_files.append(log_demand_file)
        log_delay_file = open("%s/delay_type%d.log" % (log_dir, i), "w", 1)
        log_delay_files.append(log_delay_file)
        log_throughput_file = open("%s/throughput_type%d.log" % (log_dir, i), "w", 1)
        log_throughput_files.append(log_throughput_file)
        log_loss_file = open("%s/loss_type%d.log" % (log_dir, i), "w", 1)
        log_loss_files.append(log_loss_file)
    log_globalrwd_file = open("%s/globalrwd.log" % (log_dir), "w", 1)
    log_circle_file = open("%s/circle.log" % (log_dir), "w", 1)

    # building model
    actor_critics = []
    agents = []
    rollouts = []
    # for parameter sharing
    actor_critic = Policy(node_state_dim, num_node, num_type, base_kwargs={})
    if model_load_path != None:
            actor_critic.load_state_dict(torch.load("%s/agent%d.pth" % (model_load_path, 0), map_location='cpu')) #gpu data to cpu
            #actor_critic.load_state_dict(torch.load("%s/agent%d.pth" % (model_load_path, 0))) # gpu->gpu cpu->cpu
    
    for i in range(num_agent):
        actor_critic.to(device)
        actor_critics.append(actor_critic)

        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                                    args.entropy_coef, lr=args.lr,
                                    eps=args.eps, alpha=args.alpha,
                                    max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                            args.value_loss_coef, args.entropy_coef, lr=args.lr,
                            eps=args.eps,
                            max_grad_norm=args.max_grad_norm)
        else:
            raise NotImplementedError
        agents.append(agent)

        rollout = RolloutStorage(args.num_pretrain_steps, action_spaces[i], node_state_dim, 1, num_node)
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].set_graph(torch.tensor(edge_indexs[agent_to_node[i]], dtype=torch.long).t().contiguous(), torch.tensor(adj_masks[agent_to_node[i]]), agent_to_node[i])
        rollouts[i].to(device)
        
        
    # Pre training
    #time_costs = []
    #temp_log_file = open("temp1.log", "w", 1)
    rtype = request.rtype
    for i in range(args.num_pretrain_epochs):
        for j in range(args.num_pretrain_steps):
            # interact with the environment
            with torch.no_grad():
                values = [None] * num_agent
                actions = [None] * num_agent
                action_log_probs = [None] * num_agent
                condition_states = [None] * num_agent
                
                # generate routing action route by route
                curr_path = [0] * num_node
                agents_flag = [0] * num_agent
                curr_agent, path = envs.first_agent()
                
                while curr_agent != None and agents_flag[curr_agent] != 1:
                    for k in path:
                        curr_path[k] = 1
                    agents_flag[curr_agent] = 1
                    
                    # curr_path indicate current passed nodes, may be not a simple path
                    # for example agent1->node0->agent1->node0->agent2
                    condition_state = torch.tensor(curr_path, dtype=torch.float32).unsqueeze(-1).to(device)
                    edge_index = rollouts[curr_agent].edge_index
                    x = rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0).to(device)
                    inputs = Data(x=x, edge_index=edge_index)
                    adj_mask = rollouts[curr_agent].adj_mask

                    
                    #start = time.time()
                    value, action, action_log_prob = actor_critics[curr_agent].act(
                            inputs, condition_state.unsqueeze(0), agent_to_node[curr_agent], torch.tensor([rtype]).to(device), adj_mask)        
                    #end = time.time()
                    #time_costs.append(end - start)

                    values[curr_agent] = value
                    actions[curr_agent] = action
                    action_log_probs[curr_agent] = action_log_prob
                    condition_states[curr_agent] = condition_state
                    curr_agent, path = envs.next_agent(curr_agent, action)
                # nodes not on the path's policy gradients will be zeroed when training
                condition_state = torch.tensor([0] * num_node, dtype=torch.float32).unsqueeze(-1).to(device)
                for k in range(num_agent):
                    if agents_flag[k] != 1:
                        edge_index = rollouts[k].edge_index
                        x = rollouts[k].obs[rollouts[k].step].unsqueeze(0).to(device)
                        inputs = Data(x=x, edge_index=edge_index)
                        adj_mask = rollouts[k].adj_mask
                        value, action, action_log_prob = actor_critics[k].act(
                                inputs, condition_state.unsqueeze(0), agent_to_node[k], torch.tensor([rtype]).to(device), adj_mask)
                    
                        values[k] = value
                        actions[k] = action
                        action_log_probs[k] = action_log_prob
                        condition_states[k] = condition_state

            gfactors = [0.] * num_agent
            obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, globalrwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
            
            print(delta_dist, file=log_dist_files[rtype])
            print(delta_demand, file=log_demand_files[rtype])
            print(globalrwd, file=log_globalrwd_file)
            print(circle_flag, file=log_circle_file)
            
            for k in range(num_agent):
                masks = torch.tensor([1.])
                rollouts[k].insert(obses[k], condition_states[k], actions[k].squeeze(0), action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks, torch.tensor([rtype]))

        for k in range(num_agent):
            # update learning rate (Note: DRL-OR-S only supports PPO algorithm)
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(agents[k].optimizer, j, args.num_env_steps, args.lr)
            if args.algo == 'ppo' and args.use_linear_clip_decay:
                agents[k].clip_param = args.clip_param  * (1 - i / float(args.num_pretrain_epochs))

            # update model param
            with torch.no_grad(): 
                condition_state = torch.tensor([0] * num_node, dtype=torch.float32).unsqueeze(-1).to(device)
                edge_index = rollouts[k].edge_index
                x = rollouts[k].obs[-1].unsqueeze(0)
                inputs = Data(x=x, edge_index=edge_index)
                next_value = actor_critics[k].get_value(inputs,
                                            condition_state.unsqueeze(0), agent_to_node[k], rollouts[k].type_indexs[-1]).detach()

                rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            value_loss, action_loss, dist_entropy = agents[k].update(rollouts[k])

            rollouts[k].after_update()
    #print("avg time cost:", sum(time_costs) / len(time_costs))
    

    # training episode
    if args.mode == 'train':
        deterministic = False
    else:
        deterministic = True

    request, obses = envs.reset() 
    # update rollouts(num-steps)
    rollouts = []
    for i in range(num_agent):
        rollout = RolloutStorage(args.num_pretrain_steps, action_spaces[i], node_state_dim, 1, num_node)
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].set_graph(torch.tensor(edge_indexs[agent_to_node[i]], dtype=torch.long).t().contiguous(), torch.tensor(adj_masks[agent_to_node[i]]), agent_to_node[i])
        rollouts[i].to(device)
    
    # update adam optimizer
    for k in range(num_agent):
        agents[k].reset_optimizer()
    
    # training
    time_sum = 0
    rtype = request.rtype
    for j in range(args.num_env_steps):
        
        
        # changing demand for training; applied in DRL-OR-S to improve the generalization of the trained model
        '''
        if args.mode == 'train' and j % 12800 == 0:
            envs.change_env('demand_change')
        '''

        # interact with the environment
        with torch.no_grad(): 
            values = [None] * num_agent
            actions = [None] * num_agent
            action_log_probs = [None] * num_agent
            condition_states = [None] * num_agent

            # generate routing action route by route
            curr_path = [0] * num_node
            agents_flag = [0] * num_agent
            curr_agent, path = envs.first_agent()
            
            while curr_agent != None and agents_flag[curr_agent] != 1:
                for k in path:
                    curr_path[k] = 1
                agents_flag[curr_agent] = 1
                
                condition_state = torch.tensor(curr_path, dtype=torch.float32).unsqueeze(-1).to(device)
                edge_index = rollouts[curr_agent].edge_index
                x = rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0).to(device)
                inputs = Data(x=x, edge_index=edge_index)
                adj_mask = rollouts[curr_agent].adj_mask

                start = time.time()
                value, action, action_log_prob = actor_critics[curr_agent].act(
                        inputs, condition_state.unsqueeze(0), agent_to_node[curr_agent], torch.tensor([rtype]).to(device), adj_mask, deterministic=deterministic)
                end = time.time()
                time_sum += end - start
                
                values[curr_agent] = value
                actions[curr_agent] = action
                action_log_probs[curr_agent] = action_log_prob
                condition_states[curr_agent] = condition_state
                curr_agent, path = envs.next_agent(curr_agent, action)
                
            # nodes not on the path's policy gradients will be zeroed when training
            condition_state = torch.tensor([0] * num_node, dtype=torch.float32).unsqueeze(-1).to(device)
            for k in range(num_agent):
                if agents_flag[k] != 1:
                    edge_index = rollouts[k].edge_index
                    x = rollouts[k].obs[rollouts[k].step].unsqueeze(0).to(device)
                    inputs = Data(x=x, edge_index=edge_index)
                    adj_mask = rollouts[k].adj_mask
                    
                    start = time.time()
                    value, action, action_log_prob = actor_critics[k].act(
                            inputs, condition_state.unsqueeze(0), agent_to_node[k], torch.tensor([rtype]).to(device), adj_mask, deterministic=deterministic)
                    end = time.time()
                    time_sum += end - start

                    values[k] = value
                    actions[k] = action
                    action_log_probs[k] = action_log_prob
                    condition_states[k] = condition_state
            
        gfactors = [1.] * num_agent

        obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, globalrwd, delay, throughput_rate, loss_rate = envs.step(actions, gfactors)
        print(delta_dist, file=log_dist_files[rtype])
        print(delta_demand, file=log_demand_files[rtype])
        print(delay, file=log_delay_files[rtype])
        print(throughput_rate, file=log_throughput_files[rtype])
        print(loss_rate, file=log_loss_files[rtype])
        print(globalrwd, file=log_globalrwd_file)
        print(circle_flag, file=log_circle_file)

        if args.mode == 'test':
            agent_to_node, edge_indexs, adj_masks = envs.get_topo_info()
            for k in range(num_agent):
                rollouts[k].set_graph(torch.tensor(edge_indexs[agent_to_node[k]], dtype=torch.long).t().contiguous().to(device), torch.tensor(adj_masks[agent_to_node[k]]).to(device), agent_to_node[k])
    
        
        for k in range(num_agent):
            if agents_flag[k] == 1:
                masks = torch.tensor([1.])
            else:
                masks = torch.tensor([0.])
            rollouts[k].insert(obses[k], condition_states[k], actions[k].squeeze(0), action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks, torch.tensor([rtype]))

            # update model param
            if rollouts[k].step == 0:
                # update learning rate
                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    utils.update_linear_schedule(agents[k].optimizer, j, args.num_env_steps, args.lr)
                if args.algo == 'ppo' and args.use_linear_clip_decay:
                    agents[k].clip_param = args.clip_param  * (1 - j / float(args.num_env_steps))
                
                # update actor and critic
                
                with torch.no_grad(): 
                    condition_state = torch.tensor([0] * num_node, dtype=torch.float32).unsqueeze(-1).to(device)
                    edge_index = rollouts[k].edge_index
                    x = rollouts[k].obs[-1].unsqueeze(0)
                    inputs = Data(x=x, edge_index=edge_index)
                    next_value = actor_critics[k].get_value(inputs,
                                                condition_state.unsqueeze(0), agent_to_node[k], rollouts[k].type_indexs[-1]).detach()

                    rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
                if args.mode == "train":
                    value_loss, action_loss, dist_entropy = agents[k].update(rollouts[k])

                rollouts[k].after_update()
                
        
        if j % ckpt_step == 0:
            if model_save_path != None:
                save_dir = os.path.expanduser(model_save_path)
                utils.cleanup_log_dir(save_dir)
                for i in range(num_agent):
                    torch.save(actor_critics[i].state_dict(), "%s/agent%d.pth" % (model_save_path, i))

    print("average time(s) per agent:", time_sum / args.num_env_steps / num_agent)

    if model_save_path != None:
        save_dir = os.path.expanduser(model_save_path)
        utils.cleanup_log_dir(save_dir)
        for i in range(num_agent):
            torch.save(actor_critics[i].state_dict(), "%s/agent%d.pth" % (model_save_path, i))
    
    

if __name__ == "__main__":
    total_start = time.time()
    main()
    total_end =time.time()
    print("Whole training time:", total_end - total_start)
