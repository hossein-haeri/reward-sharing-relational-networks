#!/bin/bash
cd maddpg/experiments/

python train_two_agent.py --exp-name two-agent_wsm_fully-connected_slow_01 &
python train_two_agent.py --exp-name two-agent_wsm_fully-connected_slow_02 &
python train_two_agent.py --exp-name two-agent_wsm_fully-connected_slow_03 &
python train_two_agent.py --exp-name two-agent_wsm_fully-connected_slow_04 &
python train_two_agent.py --exp-name two-agent_wsm_fully-connected_slow_05 &

wait  # This will wait for all background processes to complete
