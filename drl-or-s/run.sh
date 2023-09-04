# offline training
# python3 main.py --mode train --use-gae --num-mini-batch 1 --use-linear-lr-decay --num-env-steps 3000000 --env-name Abi --log-dir ./log/test --model-save-path ./model/test --model-load-path ./model/Abi-heavyload-gcn-sharepolicy-SPRsafe-mininet-2penalty-test-Abi-dynamic-extra2   --num-pretrain-epochs 0 --simu-port 5010

# online test
python3 main.py --mode test --use-gae --num-mini-batch 1 --use-linear-lr-decay --num-env-steps 50000 --env-name Abi --log-dir ./log/test  --model-save-path ./model/test --model-load-path ./model/Abi-heavyload-gcn-sharepolicy-SPRsafe-mininet-2penalty-test-Abi-dynamic-extra2 --num-pretrain-epochs 0  --use-mininet