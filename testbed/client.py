#coding=utf-8
import socket
import sys
import time

BUFFER_SIZE = 512

def padding_bytes(x, target_len):
    clen = len(x)
    x += bytes(target_len - clen)
    return x

if __name__ == '__main__':
    server_addr = sys.argv[1]
    server_port = int(sys.argv[2])
    client_addr = sys.argv[3]
    client_port = int(sys.argv[4])
    demand = int(sys.argv[5]) # Kbps
    rtime = int(sys.argv[6]) # seconds
    rtype = int(sys.argv[7])

    time_step = int(sys.argv[8]) # deprecated now

    if rtype == 0:
        BUFFER_SIZE = 64
    else:
        BUFFER_SIZE = 512
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((client_addr, client_port))
    ind = 0
    start_time = time.time()

    last_stamp = start_time
    CSTEP = 30

    while True:
        time_stamp = time.time()
        curr_bit = ind * BUFFER_SIZE * 8
        if curr_bit < (time_stamp - start_time) * demand * 1000:
            msg = "%d;%d;" % (ind, int(time_stamp * 1000))
            msg = padding_bytes(msg.encode(), BUFFER_SIZE)
            sock.sendto(msg, (server_addr, server_port))
            
            ind += 1
            
            '''
            # logging the sender's data rate
            if ind % CSTEP == 0:
                curr_time = time.time()
                throughput = BUFFER_SIZE * 8 * CSTEP / (curr_time - last_stamp) / 1000
                last_stamp = curr_time
                
                if ind % CSTEP * 10 == 0:
                    print("client: demand:", demand, "throughput:", throughput, "ind:", ind, flush=True) 
            '''
            
        time.sleep(BUFFER_SIZE / (demand * 125) / 2)
        #time.sleep(BUFFER_SIZE / (demand * 125)) # for testing

    sock.close()
