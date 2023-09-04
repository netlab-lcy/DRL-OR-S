'''
This file generates training data for network QoS modeling algorithm
This file interact with ryu controller and mininet testbed
'''
from utils import weight_choice
import random
import numpy as np
import os
import sys
import glob
import heapq
import copy
import json
import socket

class Request():
    def __init__(self, s, t, start_time, end_time, demand, rtype):
        # use open time interval: start_time <= time < end_time
        self.s = s
        self.t = t
        self.start_time = start_time
        self.end_time = end_time
        self.demand = demand
        self.rtype = rtype # 0: latency-sensitive; 1: throughput-sensitive; 2: latency-throughput-sensitive; 3: latency-loss-sensitive
    '''
        deprecated function
    '''
    def to_json(self):
        data = {
            'src': int(self.s),
            'dst': int(self.t),
            'time': int(self.end_time - self.start_time),
            'rtype': int(self.rtype),
            'demand': int(self.demand),
                }
        return json.dumps(data)

    
    def __lt__(self, other):
        return self.end_time < other.end_time
    
    def __str__(self):
        return "s: %d t: %d\nstart_time: %d\nend_time: %d\ndemand: %d\nrtype: %d" % (self.s, self.t, self.start_time, self.end_time, self.demand, self.rtype)

class NetEnv():
    def __init__(self, args):
        self.args = args
        
        # set up communication to remote hosts(mininet host)
        MININET_HOST_IP = '127.0.0.1' # Testbed server IP
        MININET_HOST_PORT = 5000
        self.BUFFER_SIZE = 1024
        self.mininet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mininet_socket.connect((MININET_HOST_IP, MININET_HOST_PORT))
        
        CONTROLLER_IP = '127.0.0.1'
        CONTROLLER_PORT = 3999
        self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller_socket.connect((CONTROLLER_IP, CONTROLLER_PORT))
        
    
    def setup(self, toponame):
        self._time_step = 0
        self._request_heapq = []
        
        
        # init flow type and distribution
        self._type_num = 4
        self._type_dist = np.array([0.2, 0.3, 0.3, 0.2]) # TO BE CHECKED BEFORE EXPERIMENT
        
        self._load_topology(toponame)

    def reset(self):
        self._time_step = 0
        self._request_heapq = []
        self._link_usage = [([0.] * self._node_num) for i in range(self._node_num)]
        self._node_usage = [0. for i in range(self._node_num)]
        self._update_state()
    
    '''
    interact with controller and mininet to install path rules and generate request
    @retval:
        metrics of path including delay, throughput, packet loss 
    '''
    def sim_interact(self, request, path):
        # install path in controller
        data_js = {}
        data_js['path'] = path
        data_js['ipv4_src'] = "10.0.0.%d" % (request.s + 1)
        data_js['ipv4_dst'] = "10.0.0.%d" % (request.t + 1)
        data_js['src_port'] = self._time_step % 10000 + 10000
        data_js['dst_port'] = self._time_step % 10000 + 10000 
        msg = json.dumps(data_js)
        self.controller_socket.send(msg.encode())
        self.controller_socket.recv(self.BUFFER_SIZE)
        
        # communicate to testbed
        data_js = {}
        data_js['src'] = int(request.s)
        data_js['dst'] = int(request.t)
        data_js['src_port'] = int(self._time_step % 10000 + 10000)
        data_js['dst_port'] = int(self._time_step % 10000 + 10000) 
        data_js['rtype'] = int(request.rtype)
        data_js['demand'] = int(request.demand)
        data_js['rtime'] = int(request.end_time - request.start_time)
        msg = json.dumps(data_js)
        self.mininet_socket.send(msg.encode())
        # get the feedback
        msg = self.mininet_socket.recv(self.BUFFER_SIZE)
        data_js = json.loads(msg)
        return data_js
        

    '''
    generate requests and update the state of environment
    '''
    def _update_state(self):
        # update env request heapq
        while len(self._request_heapq) > 0 and self._request_heapq[0].end_time <= self._time_step:
            request = heapq.heappop(self._request_heapq)
            path = request.path
            if path != None:
                for i in range(len(path) - 1):
                    self._link_usage[path[i]][path[i + 1]] -= request.demand
                    self._node_usage[path[i]] -= request.demand
                self._node_usage[path[-1]] -= request.demand

        # generate new request
        nodelist = range(self._node_num)
        # uniform sampling
        #s, t = random.sample(nodelist, 2)
        # sampling according to demand matrix
        ind = weight_choice(self._demand_matrix)
        s = ind // self._node_num
        t = ind % self._node_num
        
        start_time = self._time_step
        rtype = np.random.choice(list(range(self._type_num)), p=self._type_dist)
        demand = random.choice(self._request_demands[rtype])
        end_time = start_time + random.choice(self._request_times[rtype])
        print("start_time:", start_time, "end_time:", end_time)

        self._request = Request(s, t, start_time, end_time, demand, rtype)
        heapq.heappush(self._request_heapq, self._request)
        

    '''
    load the topo and setup the environment
    '''
    def _load_topology(self, toponame):
        data_path = "../topology/"
        topofile = open(data_path + toponame + "/" + "Topology.txt", "r")
        demandfile = open(data_path + toponame + "/" + 'TM.txt', "r")
        self._demand_matrix = list(map(int, demandfile.readline().split()))
        # the input file is undirected graph while here we use directed graph
        # node id for input file indexed from 1 while here from 0
        self._node_num, edge_num = list(map(int, topofile.readline().split()))
        self._edge_num = edge_num * 2
        self._observation_spaces = []
        self._action_spaces = []
        
        # build the link list
        self._link_lists = [[] for i in range(self._node_num)] # neighbor for each node
        self._link_capa = [([0] * self._node_num) for i in range(self._node_num)]
        self._link_usage = [([0] * self._node_num) for i in range(self._node_num)]
        self._node_usage = [0. for i in range(self._node_num)]
        self._link_losses = [([0] * self._node_num) for i in range(self._node_num)]
        for i in range(edge_num):
            u, v, _, c, loss = list(map(int, topofile.readline().split()))
            # since node index range from 1 to n in input file
            self._link_lists[u - 1].append(v - 1)
            self._link_lists[v - 1].append(u - 1)
            # undirected graph to directed graph
            self._link_capa[u - 1][v - 1] = c
            self._link_capa[v - 1][u - 1] = c
            self._link_losses[u - 1][v - 1] = loss
            self._link_losses[v - 1][u - 1] = loss

        # calsulate shortest path distance
        self._shr_dist = []
        for i in range(self._node_num):
            self._shr_dist.append([])
            for j in range(self._node_num):
                if j == i:
                    self._shr_dist[i].append(0)
                elif j in self._link_lists[i]:
                    self._shr_dist[i].append(1)
                else:
                    self._shr_dist[i].append(1e6) # inf
        for k in range(self._node_num):
            for i in range(self._node_num):
                for j in range(self._node_num):
                    if(self._shr_dist[i][j] > self._shr_dist[i][k] + self._shr_dist[k][j]):
                        self._shr_dist[i][j] = self._shr_dist[i][k] + self._shr_dist[k][j] 
        
        # TO BE CHECKED BEFORE EXPERIMENT
        # setup flow generating step
        if toponame == "Abi":
            self._request_demands = [[100], [1500], [1500], [500]]
            self._request_times = [[50], [50], [50], [50]] # heavy load
            #self._request_times = [[10], [10], [10], [10]] # light load
            #self._request_times = [[30], [30], [30], [30]] # mid load
        elif toponame == "GEA":
            self._request_demands = [[100], [1500], [1500], [500]]
            self._request_times = [[15], [15], [15], [15]]
     
    

    '''
    calculating the Widest Path from s to t
    '''
    def calcWP(self, s, t):
        fat = [-1] * self._node_num
        WP_dist = [- 1e6] * self._node_num
        flag = [False] * self._node_num
        WP_dist[t] = 1e6
        flag[t] = True
        cur_p = t
        while flag[s] == False:
            for i in self._link_lists[cur_p]:
                #bandwidth = max(self._link_capa[i][cur_p] - self._link_usage[i][cur_p], 0)
                bandwidth = self._link_capa[i][cur_p] - self._link_usage[i][cur_p] # for testing
                if min(bandwidth, WP_dist[cur_p]) > WP_dist[i]:
                    WP_dist[i] = min(bandwidth, WP_dist[cur_p])
                    fat[i] = cur_p
            cur_p = -1
            for i in range(self._node_num):
                if flag[i]:
                    continue
                if cur_p == -1 or WP_dist[i] > WP_dist[cur_p]:
                    cur_p = i
            flag[cur_p] = True
    
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            path.append(fat[path[cur_p]])
            cur_p += 1
        return path
    
    
    '''
    calculating the Shortest Path from s to t
    '''
    def calcSHR(self, s, t):
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            tmp_dist = 1e6
            next_hop = None
            for i in self._link_lists[path[cur_p]]:
                if self._shr_dist[i][t] < tmp_dist and self._link_capa[path[cur_p]][i] > 0:
                    next_hop = i
                    tmp_dist = self._shr_dist[i][t]
            path.append(next_hop)
            cur_p += 1
        return path
    
    '''
    calculating the Bandwidth-Constrained Shortest Path
    '''
    def calcBCSHR(self, s, t, demand):
        fat = [-1] * self._node_num
        SHR_dist = [1e6] * self._node_num
        flag = [False] * self._node_num
        SHR_dist[t] = 0
        flag[t] = True
        cur_p = t
        
        while flag[s] == False:
            for i in self._link_lists[cur_p]:
                #bandwidth = max(self._link_capa[i][cur_p] - self._link_usage[i][cur_p], 0)
                bandwidth = self._link_capa[i][cur_p] - self._link_usage[i][cur_p] 
                if bandwidth >= demand and SHR_dist[i] > SHR_dist[cur_p] + 1:
                    SHR_dist[i] = SHR_dist[cur_p] + 1
                    fat[i] = cur_p
            cur_p = -1
            for i in range(self._node_num):
                if flag[i]:
                    continue
                if cur_p == -1 or SHR_dist[i] < SHR_dist[cur_p]:
                    cur_p = i
            if SHR_dist[cur_p] < 1e6:
                flag[cur_p] = True
            else:
                break
    
        if not flag[s]:
            return None
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            path.append(fat[path[cur_p]])
            cur_p += 1
        return path

    '''
    method = "SHR"/"WP"/"DS"(diff-serv)
    '''
    def step_baseline(self, method):
        link_usage = copy.deepcopy(self._link_usage)
        link_avail_bandwidth = (np.array(self._link_capa) - np.array(self._link_usage)).tolist()

        if method == "SHR":
            path = self.calcSHR(self._request.s, self._request.t)
        elif method == "WP":
            path = self.calcWP(self._request.s, self._request.t)
        elif method == "DS":
            if self._request.rtype == 0 or self._request.rtype == 3:
                path = self.calcSHR(self._request.s, self._request.t)
            elif self._request.rtype == 1:
                path = self.calcWP(self._request.s, self._request.t)
            else:
                path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
                if path == None:
                    path = self.calcWP(self._request.s, self._request.t)
        elif method == 'QoS':
            path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
            if path == None:
                path = self.calcWP(self._request.s, self._request.t)
        else:
            raise NotImplementedError
        self._request.path = copy.copy(path)

        # update link usage according to the selected path 
        capacity = 1e9
        for i in range(len(path) - 1):
            capacity = min(capacity, self._link_capa[path[i]][path[i + 1]] - self._link_usage[path[i]][path[i + 1]])
            self._link_usage[path[i]][path[i + 1]] += self._request.demand
            self._node_usage[path[i]] += self._request.demand
        self._node_usage[path[-1]] += self._request.demand
        count = len(path) - 1
        
        request = copy.deepcopy(self._request)
        # install rules for path and generate service request in sim-env(ryu + mininet)
        ret_data = self.sim_interact(self._request, path)
        delay = ret_data['delay']
        throughput = ret_data['throughput']
        loss_rate = ret_data['loss']
        print("path:", path, "capacity:", capacity, 'delay:', delay, 'throughput:', throughput, 'loss_rate:', loss_rate) 
        # generate new state
        self._time_step += 1
        self._update_state()
        return self._node_usage, self._link_usage, link_avail_bandwidth, request, path, delay, throughput, loss_rate


if __name__ == "__main__":
    toponame = sys.argv[1]
    method = sys.argv[2] # baseline method can be SHR | WP | DS | QoS
    num_step = int(sys.argv[3])

    # setup env
    args = None 
    envs = NetEnv(args) 
    envs.setup(toponame) 
    envs.reset()
    
    # open log file
    log_dir = "./log/%s_%s_%d_test/" % (toponame, method, num_step)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.log'))
        for f in files:
            os.remove(f)
    
    trace_file = open("%s/trace.data" % (log_dir), "w", 1)

    for i in range(num_step):
        print("step:", i)
        node_usage, link_usage, link_avail_bandwidth, request, path, delay, throughput, loss_rate = envs.step_baseline(method)
        
        data = {
                'node_usage': node_usage,
                'link_usage': link_usage,
                'link_avail_bandwidth': link_avail_bandwidth,
                'demand': request.demand,
                'rtype': int(request.rtype),
                'path': path,
                'delay': delay,
                "throughput": throughput,
                "loss_rate": loss_rate}
        data_str = json.dumps(data)
        print(data_str, file=trace_file)
        
