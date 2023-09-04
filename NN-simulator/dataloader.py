import pandas as pd
import numpy as np
import json
import copy

'''
    data_trace: traffic traffice
    data_topology: network topology
    n_nodes: number of nodes
    n_links: number of links
    drl_agents: controllable nodes for drl agents
'''

def dataloader(topo_name, trace_paths=[]):
    
    data = {}
    
    if trace_paths != []:
        data_trace_all = []
        ind = 0
        for trace_path in trace_paths:
            data_trace = pd.read_csv(trace_path, header=None,
                                        sep='\n').values.tolist()
            for i in range(len(data_trace)):
                data_trace[i] = json.loads(data_trace[i][0])
                data_trace[i]["id"]=ind
                ind += 1
            data_trace_all += copy.deepcopy(data_trace)
        data["trace"] = data_trace_all
    

    data_topology = []
    f = open("../topology/%s/Topology.txt" % (topo_name), "r")
    line = f.readline()
    line = line.split(" ")
    n_nodes = int(line[0])
    n_links = int(line[1])
    
    for i in range(n_links):
        line = f.readline()
        line = line.split(" ")
        link = {}
        link["node_a"] = int(line[0])-1
        link["node_b"] = int(line[1])-1
        link["weight"] = float(line[2])
        link["capacity"] = float(line[3])
        link["loss"] = float(line[4]) / 100
        data_topology.append(link)
        
        link = {}
        link["node_a"] = int(line[1])-1
        link["node_b"] = int(line[0])-1
        link["weight"] = float(line[2])
        link["capacity"] = float(line[3])
        link["loss"] = float(line[4]) / 100
        data_topology.append(link)
        
        
    line = f.readline()
    line = line.split(" ")
    drl_agents = []
    for i in range(len(line)):
        drl_agents.append(bool(int(line[i])))
    f.close()

    
    data["topology"] = data_topology
    data["n_links"] = n_links * 2
    data["n_nodes"] = n_nodes
    data["drl_agents"] = drl_agents
    return data