#!/bin/bash
cd maddpg/experiments/


num_runs=3
num_episodes=250000

for i in {1..$num_runs}
do
python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WSM --network fully-connected --num-episdoes $num_episodes --exp-name $i &
done

for i in {1..$num_runs}
do
python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WPM --network fully-connected --num-episdoes $num_episodes --exp-name $i &
done

for i in {1..$num_runs}
do
python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type MinMax --network fully-connected --num-episdoes $num_episodes --exp-name $i &
done

wait  # This will wait for all background processes to complete
