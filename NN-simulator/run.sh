# train
# python3 main.py --mode train --log-dir test --use-cuda --train-data-dir ../data_generation/log/Abi_SHR_100000_final-loss-heavyload-largequeue --test-data-dir ../data_generation/log/Abi_SHR_10000_final-loss-heavyload-largequeue

# test
# python3 main.py --mode test --log-dir test --use-cuda --test-data-dir ../data_generation/log/Abi_SHR_10000_final-loss-heavyload-largequeue

# simulate
# log-dir(model): for Abilene: final-loss-100queuesize, for GEANT: final-loss-GEA-100queuesize
python3 main.py --mode simulator --log-dir final-loss-100queuesize --use-cuda --topo Abi --simu-port 5010