# A-Crab: Actor-Critic Regularized by Average Bellman error

This repository contains the code to reproduce the experimental results of A-Crab algorithm in the paper [<em>Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning </em>](https://arxiv.org/abs/2301.12714) by Hanlin Zhu, Paria Rashidinejad, and Jiantao Jiao. Most of the code and instructions are adapted from and follow the logic in the [lightATAC](https://github.com/microsoft/lightATAC/tree/main) repo.

### Setup 

#### Step 1: Clone the repository.
```
git clone https://github.com/zhuhl98/ACrab.git
```
or
```
git clone git@github.com:zhuhl98/ACrab.git
```

#### Step 2: Create a conda environment.

```
conda create -n ACrab python=3.9
conda activate ACrab
cd ACrab
```

#### Step 3: To install, run 
```
pip install -e .
```

#### Step 4: It uses mujoco210, which can be installed, if needed, following the commands below.

```
bash install.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia" >> ~/.bashrc
source ~/.bashrc
```

#### Step 5: You can start the training by running, e.g.,

    python main.py --log_dir ./tmp_results --env_name hopper-medium-expert-v2 --beta 1 --C_infty 1

More instructions can be found in `main.py`, and please see the [original paper](https://arxiv.org/abs/2301.12714) for hyperparameters (e.g., `beta`, `C_infty`). The code was tested with python 3.9.


