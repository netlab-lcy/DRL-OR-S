# training with mininet (online training)
# python3 main.py --use-gae --num-mini-batch 1 --use-linear-lr-decay --num-env-steps 100000 --env-name GEA --log-dir ./log/GEA-QoSsafe-mininet  --model-save-path ./model/GEA-QoSsafe-mininet --model-load-path ./model/GEA-QoSsafe-simulator-extra2 --use-mininet --num-pretrain-epoch 0 

# training with NN-simulator (offline training)
# python3 main.py --use-gae --num-mini-batch 1 --use-linear-lr-decay --num-env-steps 1000000 --env-name GEA --log-dir ./log/GEA-QoSsafe-simulator-extra2 --model-save-path ./model/GEA-QoSsafe-simulator-extra2 --model-load-path ./model/GEA-heavyload-QoSsafe-simulator-extra  --simu-port 5020 --no-cuda  --num-pretrain-epoch 0
# python3 main.py --use-gae --num-mini-batch 1 --use-linear-lr-decay --num-env-steps 1000000 --env-name DialtelecomCz --log-dir ./log/DialtelecomCz-QoSsafe-mininet --model-save-path ./model/DialtelecomCz-QoSsafe-mininet  --use-mininet --num-pretrain-epoch 30 --no-cuda 

# baseline
# SPR
python3 net_env/simenv.py Abi SHR 50000
# LBR 
python3 net_env/simenv.py Abi WP 50000
# QoSR
python3 net_env/simenv.py Abi DS 50000
# MARL-GNN-TE
python3 net_env/simenv.py Abi WSHR 50000
