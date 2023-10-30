@echo off
cd C:\Users\Hossein_Haeri\reward-sharing-relational-networks\venv\Scripts
call activate
cd ..\..\maddpg\experiments
start python train_v0.py --exp-name athoritarian_01
start python train_v0.py --exp-name athoritarian_02
start python train_v0.py --exp-name athoritarian_03
start python train_v0.py --exp-name athoritarian_04
start python train_v0.py --exp-name athoritarian_05
start python train_v0.py --exp-name athoritarian_06
start python train_v0.py --exp-name athoritarian_07
start python train_v0.py --exp-name athoritarian_08
start python train_v0.py --exp-name athoritarian_09
start python train_v0.py --exp-name athoritarian_10
cmd /k