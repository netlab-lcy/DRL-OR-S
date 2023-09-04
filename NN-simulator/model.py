import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class QoSNet(nn.Module):
    """
    Args:
        output_units (int): Output units for the last layer.
    """

    def __init__(self, args, output_units=1):
        super(QoSNet, self).__init__()
        self.args = args
        self.output_units = output_units

        
        self.gru_rnn = nn.GRU(
            args.link_state_dim,
            args.path_state_dim,
            batch_first=True,
        )

        self.normalize = nn.BatchNorm1d(args.path_state_dim)

        readout_layers = []
        for i in range(output_units):
            readout_layer = nn.Sequential(
                nn.Linear(
                    args.path_state_dim,
                    args.readout_units,
                ),
                nn.SELU(False),
                nn.Linear(
                    args.readout_units,
                    args.readout_units,
                ),
                nn.SELU(False),
                nn.Linear(
                    args.readout_units,
                    1,
                )
            )
            readout_layers.append(readout_layer)
        self.readout_layers = nn.ModuleList(readout_layers)
        
    
    def forward(self, input):
        f_ = input
        n_links = int(f_["n_links"][0])
        path = f_["path"]
        demand = f_["demand"].float()
        link_avail_capacity = f_["link_avail_capacity"].float()
        link_capacity = f_["link_capacity"].float()
        link_loss = f_["link_loss"].float()
        hop_num = f_['hop_num'] # batch 
        bs = demand.size(0)
        max_capa = torch.max(link_capacity).detach().item()
        
        # In this version we do not take link latency into consideration since DRL-OR-S set the same latency to the links (5ms)
        # TODO: add the link latency to the link_state
        link_state = torch.cat(
            [torch.zeros(bs, n_links, self.args.link_state_dim - 4), 
            (link_avail_capacity.unsqueeze(-1)) / max_capa,
            link_capacity.unsqueeze(-1) / max_capa,
            (link_avail_capacity.unsqueeze(-1)) / link_capacity.unsqueeze(-1),
            link_loss.unsqueeze(-1),
            ],
            dim=-1
        ).to(self.args.device)
        path_state = torch.cat(
            [torch.zeros(bs, self.args.path_state_dim - 1),
            demand.unsqueeze(-1) / max_capa],
            dim=-1
        ).to(self.args.device)
        max_len = hop_num.max().item()
        
        shape = [bs, max_len, self.args.link_state_dim]
        link_inputs = torch.zeros(shape).to(self.args.device)
        for i in range(bs):
            for j in range(len(path[i])):
                id = int(path[i][j])
                link_inputs[i][j] = link_inputs[i][j] + link_state[i][id].detach()
        
        path_state = torch.unsqueeze(path_state, 0)
        outputs, path_state = self.gru_rnn(
            link_inputs,
            path_state
        )
        
        flow_features = torch.zeros(bs, self.args.path_state_dim).to(self.args.device)
        for i in range(bs):
            flow_features[i] += outputs[i][len(path[i]) - 1]

        flow_features = self.normalize(flow_features)
        ys = []
        for i in range(self.output_units):
            ys.append(self.readout_layers[i](flow_features))
        y = torch.cat(ys, dim=-1)
        return y
        
