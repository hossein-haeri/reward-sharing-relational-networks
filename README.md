Reward-Sharing Relational Networks (RSRN) in Multi-Agent Reinforcement Learning (MARL)
======================================================================================


Introduction
------------

This repository contains the Python implementation of our work on integrating 'social' interactions in Multi-Agent Reinforcement Learning (MARL) setups through a user-defined relational network. The study aims to understand the impact of agent-agent relations on emergent behaviors using the concept of Reward-Sharing Relational Networks (RSRN).

Abstract
--------

This work and the associated code are based on the paper 'Reward-Sharing Relational Networks in Multi-Agent Reinforcement Learning as a Framework for Emergent Behavior' by Hossein Haeri, Reza Ahmadzadeh, and Kshitij Jerath. If you find our work useful or use it in your research, please consider citing our paper:

https://arxiv.org/abs/2207.05886

You can find more detail on the project website: https://sites.google.com/view/marl-rsrn
https://drive.google.com/file/d/1LTxAY6wN31Quw7PeOfRqSNqlvunOlu0v/view?usp=sharing


Simulation and Scenario
-----------------------

We leverage the Multi-agent Particle Environment (MPE) for simulating an RSRN in a 3-agent MARL environment. The agents' performance under different network structures is evaluated in this setting.

Environment Details:

- Framework: Multi-agent Particle Environment (MPE)
- Agents: Modeled as physical elastic objects
- State and Action Spaces: Continuous
- Policy Optimization: Integration of Relational Network and Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
- Network Structure: 2 fully-connected 64-unit layers
- Learning Parameters: 
  - Learning Rate: 0.01
  - Batch Size: 2048
  - Discount Ratio: 0.99

Scenario Description:

Three agents aim to reach three unlabeled landmarks. Rewards are given upon reaching any landmark. This design makes the multi-agent environment intricate, thereby providing ample opportunities for emergent behaviors.

Setup & Usage
-------------

1. Dependencies: Ensure you have all the necessary dependencies installed. (Provide a `requirements.txt` for ease.)
2. Training: To train the agents, run `python train_v2.py`. Use argument --exp-name to keep track of your experiment Adjust the training parameters as required but the default parameters are the ones used in the paper.
3. Visualization: To visualize the behavior of the agents use --restore to load an already trained experiment and use --display to see the agent behaviors.




