Reward-Sharing Relational Networks (RSRN) in Multi-Agent Reinforcement Learning (MARL)
======================================================================================

Introduction
------------

This repository contains the Python implementation of our work on integrating 'social' interactions in Multi-Agent Reinforcement Learning (MARL) setups through a user-defined relational network. The study aims to understand the impact of agent-agent relations on emergent behaviors using the concept of Reward-Sharing Relational Networks (RSRN).

Abstract
--------

In this work, we integrate ‘social’ interactions into the MARL setup through a user-defined relational network and examine the effects of agent-agent relations on the rise of emergent behaviors. By combining insights from sociology and neuroscience, our proposed framework models agent relationships using the notion of RSRN. In this context, network edge weights act as a metric indicating the degree to which one agent values the success of another. We construct relational rewards based on the RSRN interaction weights to train the multi-agent system collectively through a MARL algorithm. The system's efficacy is assessed in a 3-agent scenario under varied relational network structures. Our findings reveal that reward-sharing relational networks profoundly affect the learned behaviors, with the RSRN framework leading to emergent behaviors often reflecting the sociological understanding of such networks.

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
2. Training: To train the agents, run `python train.py`. Adjust the training parameters as required.
3. Evaluation: Once trained, evaluate the agent's performance with `python evaluate.py`.
4. Visualization: Visual results are depicted in Figure 1 (Ensure there's a link or the image is embedded).

Conclusion & Future Work
------------------------

Our research underscores the profound influence of reward-sharing relational networks on emergent behaviors in MARL settings. The RSRN framework offers a platform where various relational networks yield distinct emergent behaviors, closely mirroring our sociological understanding of these networks.

Citing our Work
---------------

If you find our work useful or use it in your research, please consider citing our paper. (Provide a BibTeX citation or the relevant paper details.)

Feedback & Contributions
------------------------

We welcome feedback, issues, and pull requests. Feel free to reach out for any queries or suggestions.

License
-------

This project is licensed under the MIT License.

