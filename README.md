# Scalable Deep Reinforcement Learning-Based Online Routing for Multi-type Service Requirements

This is a Pytorch implementation of DRL-OR-S on TPDS 2023. 

## Requirements

To run DRL-OR-S

* Firstly you should install [mininet](http://mininet.org/download/)

```
git clone git://github.com/mininet/mininet
cd mininet
util/install.sh -a
```

* Then install ryu-manager and other required packages

```
pip3 install ryu
pip3 install -r requirements.txt
```
## Offline training DRL-OR-S with NN-simulator
We have uploaded the [training and testing data](https://1drv.ms/f/s!AkZtd2aHAb3nlEtwt_DuZgKPV9yt?e=kFBfg3) for NN-simulator.

To training DRL-OR-S offline

* Run NN-simulator

```
cd NN-simulator
python3 main.py --mode simulator --log-dir final-loss-100queuesize --use-cuda --topo Abi --simu-port 5010
```

* Run DRL-OR-S algorithm

```
python3 main.py --mode train --use-gae --num-mini-batch 1 --use-linear-lr-decay --num-env-steps 3000000 --env-name Abi --log-dir ./log/test --model-save-path ./model/test --model-load-path ./model/Abi-heavyload-gcn-sharepolicy-SPRsafe-mininet-2penalty-test-Abi-dynamic-extra2   --num-pretrain-epochs 0 --simu-port 5010
```

## Online testing DRL-OR-S with mininet testbed

To run DRL-OR code as an example

* Run testbed

```
cd testbed
sudo ./run.sh
```

* Run ryu controller

```
cd ryu-controller
./run.sh
```

* Run DRL-OR-S algorithm

```
cd drl-or-s
./run.sh
```

If you have any questions, please post an issue or send an email to chenyiliu9@gmail.com.

## Citation

```
@article{liu2023scalable,
  title={Scalable Deep Reinforcement Learning-Based Online Routing for Multi-Type Service Requirements},
  author={Liu, Chenyi and Wu, Pingfei and Xu, Mingwei and Yang, Yuan and Geng, Nan},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  year={2023},
  publisher={IEEE}
}
```



