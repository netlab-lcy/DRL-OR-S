import argparse

def get_arg():
    parser = argparse.ArgumentParser(description='NN-simulator')

    # Add config here
    # NN model config
    parser.add_argument('--link-state-dim', type=int, default=8, 
        help='Dimention of link state vector')
    parser.add_argument('--path-state-dim', type=int, default=64, 
        help='Dimention of path state vector')
    parser.add_argument('--readout-units', type=int, default=32,
        help='Number of readout hidden units')
    parser.add_argument('--lr', type=float, default=0.0001,
        help='learning rate')
    


    # running config
    parser.add_argument('--log-dir', default="test",
        help='name of model and log dir.')
    parser.add_argument('--mode', default="train",
        help='NN-simulator mode(train|simulator)')
    parser.add_argument('--training-epochs', type=int,default=100,
        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
        help='Sample batch size from replay buffer')
    parser.add_argument('--use-cuda', action='store_true', default=False,
        help='Use cuda to speed up algorithm.')
    parser.add_argument('--simu-port', type=int, default=5000,
        help='TCP port for simulator/mininet (default: 5000)')
    parser.add_argument('--topo', default="Abi",
        help='Topology name')
    parser.add_argument('--train-data-dir', default="../data_generation/log/Abi_SHR_100000_final-loss-heavyload-largequeue",
        help='Training data directory')
    # for GEA: GEA_SHR_100000_final-loss-midload-largequeue
    parser.add_argument('--test-data-dir', default="../data_generation/log/Abi_SHR_10000_final-loss-heavyload-largequeue",
        help='Test data directory')
    # for GEA: GEA_SHR_10000_final-loss-midload-largequeue

    args = parser.parse_args()

    return args