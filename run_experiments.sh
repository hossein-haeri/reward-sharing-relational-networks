#!/bin/bash
cd maddpg/experiments/
python train_v0.py --exp-name fully-connected_01
# python train_v0.py --exp-name fully-connected_02 &
# python train_v0.py --exp-name fully-connected_03 &
# python train_v0.py --exp-name fully-connected_04 &
# python train_v0.py --exp-name fully-connected_05 &
# python train_v0.py --exp-name fully-connected_06 &
# python train_v0.py --exp-name fully-connected_07 &
# python train_v0.py --exp-name fully-connected_08 &
# python train_v0.py --exp-name fully-connected_09 &
# python train_v0.py --exp-name fully-connected_10 &
# python train_v0.py --exp-name fully-connected_11 &
# python train_v0.py --exp-name fully-connected_12 &

wait  # This will wait for all background processes to complete
