#coding=utf-8
import socket 
import sys
import time

BUFFER_SIZE = 512
TIME_OUT = 5 

if __name__== '__main__': 
    addr = sys.argv[1]
    port = int(sys.argv[2])
    rtime = int(sys.argv[3])
    rtype = int(sys.argv[4])

    time_step = int(sys.argv[5]) # deprecated now
    
    if rtype == 0:
        BUFFER_SIZE = 64
    else:
        BUFFER_SIZE = 512

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((addr, port))
    
    ind = 0
    delay = 0
    throughput = 0
    time_stamp = time.time()
    start_time = time.time()
    if rtype == 0:
        CSTEP = 10 # small flow only need delay, small CSTEP can speed up the experiment
    else:
        CSTEP = 30
    
    ind_stamp = 0
    while True:
        try:
            sock.settimeout(TIME_OUT)
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            continue
        
        infos = str(data.decode()).split(';')[:-1] 
        delay += int(time.time() * 1000) - int(infos[1])
        throughput += BUFFER_SIZE * 8
        ind += 1
        if ind % CSTEP == 0:
            # No packet disorder in simulate environment
            # We only need the first several records, too much record (about 1000 records) will crash popen buffer and make the server killed
            if ind / CSTEP <= 10:
                print("delay: %f ms throughput: %f Kbps loss_rate: %f" % (delay / CSTEP, throughput / 1e3 / (time.time() - time_stamp), (int(infos[0]) - ind_stamp - CSTEP) / (int(infos[0]) - ind_stamp)), flush=True) # to flush the content to popen and pmonitor
            delay = 0
            throughput = 0
            time_stamp = time.time()
            ind_stamp = int(infos[0])
        
    sock.close()

