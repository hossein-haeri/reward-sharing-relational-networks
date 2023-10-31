#!/bin/bash
cd maddpg/experiments/
python train_v0.py --exp-name Tribal_01 &   # The & at the end will run the script in the background
python train_v0.py --exp-name Tribal_02 &
python train_v0.py --exp-name Tribal_03 &
python train_v0.py --exp-name Tribal_04 &
python train_v0.py --exp-name Tribal_05 &
python train_v0.py --exp-name Tribal_06 &
python train_v0.py --exp-name Tribal_07 &
python train_v0.py --exp-name Tribal_08 &
python train_v0.py --exp-name Tribal_09 &
python train_v0.py --exp-name Tribal_10 &



wait  # This will wait for all background processes to complete
