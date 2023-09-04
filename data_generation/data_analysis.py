import json

trace_file = open("./log/Abi_SHR_10000_final-loss-heavyload-largequeue/trace.data", "r")
congestions = 0
congestions_real = 0
ind = 0
for line in trace_file.readlines():
    flag1 = False
    flag2 = False
    data = json.loads(line)
    path = data['path']
    demand = data['demand']
    link_avail_bandwidth  = data['link_avail_bandwidth']
    delay = data['delay']
    throughput = data['throughput']
    loss_rate = data['loss_rate']
    node_usage = data['node_usage']
    path_node_usage = [node_usage[i] for i in path]
    capacity = 1e6
    congestion_link = None
    for i in range(len(path) - 1):
        if link_avail_bandwidth[path[i]][path[i + 1]] < capacity:
            capacity = min(capacity, link_avail_bandwidth[path[i]][path[i + 1]])
            congestion_link = (path[i], path[i + 1])
       
    if delay > (len(path) - 1) * 5 * 1.2:
        congestions_real += 1
        flag1 = True
    '''
    if congestion_link == (0, 4) or congestion_link == (4, 0):
        congestion_link_capacity = 2480
    else:
        congestion_link_capacity = 9920
    '''
    if (capacity - demand)  < 0:
        congestions += 1
        flag2 = True
    

    if flag1 != flag2:
        print("False ind:", ind, "capacity:", capacity, "demand:", demand, "delay:", delay, "throughput:", throughput, "loss_rate:", loss_rate, "capacity flag:", flag2, "delay flag:", flag1, "capacity left:", capacity - demand, "path:", path, "node_usage:", path_node_usage, "max_node_usage:", max(path_node_usage), "congestion_link:", congestion_link)
    '''
    if flag1:
        print("ind:", ind, "capacity:", capacity, "demand:", demand, "delay:", delay, "throughput:", throughput, "loss_rate:", loss_rate, "capacity flag:", flag2, "delay flag:", flag1, "capacity left:", capacity - demand)
    #if abs(throughput / demand - 1) > 0.1:
    #    print("Throughput anomaly!", "ind:", ind, "capacity:", capacity, "demand:", demand, "delay:", delay, "throughput:", throughput, "loss_rate:", loss_rate, "capacity flag:", flag2, "delay flag:", flag1, "capacity left:", capacity - demand)
    
    if capacity - demand < 1000 and capacity - demand > 0:
        print("suspicious! ind:", ind, "capacity:", capacity, "demand:", demand, "delay:", delay, "throughput:", throughput, "loss_rate:", loss_rate, "capacity flag:", flag2, "delay flag:", flag1, "capacity left:", capacity - demand, "path:", path, "node_usage:", path_node_usage, "max_node_usage:", max(path_node_usage), "congestion_link:", congestion_link)
    '''
    ind += 1
print("congestions:", congestions, "congestions_real", congestions_real)
