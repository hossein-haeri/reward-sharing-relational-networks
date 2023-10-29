@echo off
cd C:\Users\Hossein_Haeri\reward-sharing-relational-networks\venv\Scripts
call activate
cd ..\..\maddpg\experiments
start python train_v0.py --exp-name fully-connected_01
start python train_v0.py --exp-name fully-connected_02
start python train_v0.py --exp-name fully-connected_03
start python train_v0.py --exp-name fully-connected_04
start python train_v0.py --exp-name fully-connected_05
start python train_v0.py --exp-name fully-connected_06
start python train_v0.py --exp-name fully-connected_07
start python train_v0.py --exp-name fully-connected_08
start python train_v0.py --exp-name fully-connected_09
start python train_v0.py --exp-name fully-connected_10
cmd /k