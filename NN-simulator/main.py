import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from model import QoSNet
from config.arguments import get_arg
from torch.utils.tensorboard import SummaryWriter
import random
import csv
import codecs
import json
import socket
from utils import cleanup_dir
import time


def data_write_csv(file_name, datas):  
    file_csv = codecs.open(file_name, 'w+', 'utf-8') 
    writer = csv.writer(file_csv)
    writer.writerow(["id","real_value","predict_value", "hop_num", "avail_capacity", "demand"])
    for data in datas:
        writer.writerow([data["id"],data["real"],data["predict"],data["hop_num"], data["avail_capacity"], data["demand"]])

def data_generator(data, batch_size, shuffle=False):
    node_matrix=np.zeros((data["n_nodes"], data["n_nodes"]), dtype=np.int64)
    node_matrix=node_matrix - 1
    for l in range(len(data["topology"])):
        node_matrix[data["topology"][l]["node_a"]][data["topology"][l]["node_b"]] = l

    if shuffle:
        random.shuffle(data["trace"])
    
    batch_num = int(len(data["trace"]) / batch_size)
    for j in range(batch_num):
        input = {}
        input["n_links"] = []
        input["demand"] = []
        input["path"] = []
        input["link_avail_capacity"] = []
        input["link_capacity"] = []
        input["link_loss"] = []
        input["throughput"] = []
        input["thrpt_ratio"] = []
        input["delay"] = []
        input["loss_rate"] = []
        input["hop_num"] = []
        input["id"] = []
        for k in range(batch_size):
            input["id"].append(data["trace"][j * batch_size + k]["id"])
            input["n_links"].append(data["n_links"])
            input["demand"].append(data["trace"][j * batch_size + k]["demand"])
            node_sequence=data["trace"][j * batch_size + k]["path"]
            path = []
            if (len(node_sequence) >= 2):
                for l in range(len(node_sequence) - 1):
                    path.append(node_matrix[node_sequence[l]][node_sequence[l + 1]])
            input["path"].append(path)
            input["hop_num"].append(len(path))
            link_avail_capacity = []
            link_capacity = []
            link_loss = []
            link_avail_capacity_matrix = np.array(data["trace"][j * batch_size + k]["link_avail_bandwidth"])
            for l in range(len(data["topology"])):
                link_avail_capacity.append(
                    float(link_avail_capacity_matrix[data["topology"][l]["node_a"]][data["topology"][l]["node_b"]]))
                link_capacity.append(data["topology"][l]["capacity"])
                link_loss.append(data['topology'][l]['loss'])
            
            input["link_avail_capacity"].append(link_avail_capacity)
            input["link_capacity"].append(link_capacity)
            input["link_loss"].append(link_loss)
            input["delay"].append(data["trace"][j * batch_size + k]["delay"])
            input["throughput"].append(data["trace"][j * batch_size + k]["throughput"])
            input["loss_rate"].append(data["trace"][j * batch_size + k]["loss_rate"])
            input["thrpt_ratio"].append(data["trace"][j * batch_size + k]["throughput"] / data["trace"][j * batch_size + k]["demand"])
        input["demand"] = torch.from_numpy(np.array(input["demand"]))
        input["delay"] = torch.from_numpy(np.array(input["delay"]))
        input["throughput"] = torch.from_numpy(np.array(input["throughput"]))
        input["thrpt_ratio"] = torch.from_numpy(np.array(input["thrpt_ratio"]))
        input["loss_rate"] = torch.from_numpy(np.array(input["loss_rate"]))
        input["link_avail_capacity"] = torch.from_numpy(np.array(input["link_avail_capacity"]))
        input["link_capacity"] = torch.from_numpy(np.array(input["link_capacity"]))
        input["hop_num"] = torch.from_numpy(np.array(input["hop_num"]))
        input["link_loss"] = torch.from_numpy(np.array(input["link_loss"]))
        yield input



def model_train(args, data, metrics):
    model = QoSNet(args, output_units=len(metrics)).to(args.device)

    optimizer=optim.Adam(model.parameters())

    loss_fn = nn.SmoothL1Loss(reduction='mean')


    train_batch_size = args.batch_size
    batch_num = int(len(data["trace"]) / train_batch_size)

    for i in range(args.training_epochs):
        train_data_gen = data_generator(data, train_batch_size, True)
        step = 0
        for sample in train_data_gen:
            step += 1
            model.train()
            labels = []
            for metric in metrics:
                labels.append(sample[metric].unsqueeze(-1))
            target = torch.cat(labels, dim=-1).float().to(args.device)
            output = model(input=sample).float()

            loss = loss_fn(output, target)
            print("traing: epoch" + str(i + 1) + "\t" + "step:" + str(step + 1) + "\t" + "loss:" + str(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer_training.add_scalar('training loss', loss.item(), i * batch_num + step)
            
    print('Finished Model Training')
    cleanup_dir("./model/%s" % (args.log_dir))
    torch.save(model.state_dict(), './model/%s/model.pt' % args.log_dir)


def predict_model(args, data, metrics):
    p_model=QoSNet(args, output_units=len(metrics)).to(args.device)
    predict_batch_size = 1
    
    test_data_gen = data_generator(data, predict_batch_size, False)
    results = []
    p_model.load_state_dict(torch.load('./model/%s/model.pt' % (args.log_dir)))
    p_model.eval()
    start = time.time()
    for sample in test_data_gen:
        result={}
        labels = []
        for metric in metrics:
            labels.append(sample[metric].detach()[0].item())
        output = p_model(sample).float()
        result["id"]=int(sample["id"][0])
        result["real"]=str(labels)
        result["predict"]=str(output[0].cpu().detach().numpy().tolist())
        result["hop_num"] = int(len(sample['path'][0]))
        avail_capacity  = 1e6
        for i in sample['path'][0]:
            avail_capacity = min(avail_capacity, sample['link_avail_capacity'].numpy().tolist()[0][i])
        result['avail_capacity'] = avail_capacity
        result["demand"] = float(sample["demand"][0])
        results.append(result)
    end = time.time()
    print('Finished Model Predicting, time consuming', end - start)
    data_write_csv("./log/%s/predict_result_%s.csv" % (args.log_dir, args.topo), results)
    
def qos_inference(model, request, topoinfo):
    node_matrix=np.zeros((topoinfo["n_nodes"], topoinfo["n_nodes"]), dtype=np.int64)
    node_matrix=node_matrix - 1
    for l in range(len(topoinfo["topology"])):
        node_matrix[topoinfo["topology"][l]["node_a"]][topoinfo["topology"][l]["node_b"]] = l
    
    input = {}
    input["n_links"] = [topoinfo["n_links"]]
    input["demand"] = [request['demand']]
    node_sequence=request["path"]
    path = []
    if (len(node_sequence) >= 2):
        for l in range(len(node_sequence) - 1):
            path.append(node_matrix[node_sequence[l]][node_sequence[l + 1]])
    input["path"] = [path]
    link_avail_capacity = []
    link_capacity = []
    link_loss = []
    link_avail_capacity_matrix = np.array(request["link_capacity"]) - np.array(request["link_usage"])
    for l in range(len(topoinfo["topology"])):
        link_avail_capacity.append(
            float(link_avail_capacity_matrix[topoinfo["topology"][l]["node_a"]][topoinfo["topology"][l]["node_b"]]))
        link_capacity.append(topoinfo["topology"][l]["capacity"])
        link_loss.append(topoinfo['topology'][l]['loss'])
    input["link_avail_capacity"] = [link_avail_capacity]
    input["link_capacity"] = [link_capacity]
    input["link_loss"] = [link_loss]
    input["hop_num"] = [len(path)]

    input["demand"] = torch.from_numpy(np.array(input["demand"]))
    input["link_avail_capacity"] = torch.from_numpy(np.array(input["link_avail_capacity"]))
    input["link_capacity"] = torch.from_numpy(np.array(input["link_capacity"]))
    input["hop_num"] = torch.from_numpy(np.array(input["hop_num"]))
    input["link_loss"] = torch.from_numpy(np.array(input["link_loss"]))

    with torch.no_grad():
        output = model(input).float()
        predict = output[0].detach().cpu().numpy().tolist()
    return predict[0], predict[1] * request['demand'], predict[2]
    

if __name__ == "__main__":
    metrics = ['delay', 'thrpt_ratio', 'loss_rate']
    args = get_arg()
    args.device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'

    if args.mode == "train":
        writer_training = SummaryWriter('runs/training')
        
        train_data = dataloader(args.topo, [args.train_data_dir+"/trace.data"]) 
        test_data = dataloader(args.topo, [args.test_data_dir+"/trace.data"]) 

        model_train(args, train_data, metrics)
        
        cleanup_dir("./log/%s" % (args.log_dir))
        predict_model(args, test_data, metrics)
    elif args.mode == "test":
        test_data = dataloader(args.topo, [args.test_data_dir+"/trace.data"])
        predict_model(args, test_data, metrics)
    elif args.mode == "simulator":
        model = QoSNet(args, output_units=len(metrics)).to(args.device)
        # model.load_state_dict(torch.load('./model/%s/model.pt' % (args.log_dir), map_location=torch.device('cpu')))
        model.load_state_dict(torch.load('./model/%s/model.pt' % (args.log_dir)))
        model.eval()
        
        topo_info = dataloader(args.topo)

        TCP_IP = "127.0.0.1"
        TCP_PORT = args.simu_port
        BUFFER_SIZE = 8192
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
    
        conn, addr = s.accept()
        print('Connection address:', addr)
        # receive instruction from sim_env.py and generate request and send results
        time_step = 0
        while True:
            try:
                msg = conn.recv(BUFFER_SIZE)
            except:
                socket.close()
                break
            print("msg:", msg)

            data_js = json.loads(msg)
            delay, throughput, loss = qos_inference(model, data_js, topo_info) 
            delay = max(0, delay)
            throughput = max(0, throughput)
            loss = max(0, loss)
        
            ret = {
                    'delay': delay,
                    'throughput': throughput,
                    'loss': loss,}
        
            msg = json.dumps(ret)
            conn.send(msg.encode())
            time_step += 1
