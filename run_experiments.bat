@echo off
cd C:\Users\Hossein_Haeri\reward-sharing-relational-networks\venv\Scripts
call activate
cd ..\..\maddpg\experiments
start python train_v0.py --exp-name fully-connected_01
@REM start python train_v0.py --exp-name fully-connected_02
@REM start python train_v0.py --exp-name fully-connected_03
@REM start python train_v0.py --exp-name fully-connected_04
@REM start python train_v0.py --exp-name fully-connected_05
@REM start python train_v0.py --exp-name fully-connected_06
@REM start python train_v0.py --exp-name fully-connected_07
@REM start python train_v0.py --exp-name fully-connected_08
@REM start python train_v0.py --exp-name fully-connected_09
@REM start python train_v0.py --exp-name fully-connected_10
@REM start python train_v0.py --exp-name fully-connected_11
@REM start python train_v0.py --exp-name fully-connected_12
cmd /k