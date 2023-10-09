# A-Crab: Actor-Critic Regularized by Average Bellman error

This repository contains the code to reproduce the experimental results of A-Crab algorithm in the paper [<em>Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning </em>](https://arxiv.org/abs/2301.12714) by Hanlin Zhu, Paria Rashidinejad, and Jiantao Jiao. Most of the code and instructions are adapted from and follow the logic in the [lightATAC](https://github.com/microsoft/lightATAC/tree/main) repo.

### Setup 

#### Clone the repository and create a conda environment.
```
git clone https://github.com/microsoft/ATAC.git
conda create -n acrab python=3.9
cd ACrab
```

To install, run 
```pip install -e .
```

It uses mujoco210, which can be installed, if needed, following the commands below.

```
bash install.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia" >> ~/.bashrc
source ~/.bashrc
```

Then you can start the training by running, e.g.,

    python main.py --log_dir ./tmp_results --env_name hopper-medium-expert-v2 --beta 1 --C_infty 1

More instructions can be found in `main.py`, and please see the [original paper](https://arxiv.org/abs/2301.12714) for hyperparameters (e.g., `beta`, `C_infty`). The code was tested with python 3.9.


