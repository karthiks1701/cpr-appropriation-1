import sys
sys.path.append('../')
import numpy as np
import gym
import ray
import matplotlib.pyplot as plt
from IPython import display

from src import rllib

n_agents = 1
grid_width = 25 
grid_height = 7
max_episodes = 1000
num_workers = 1
seed = 42
tagging_ability = True
gifting_mechanism = 0
rllib_log_dir = "../rllib_logs/"
wandb_api_key = "18638505c190d67f1040f6945fe54ab128a99dde"
wandb_project = "cpr-appropriation"

ray.shutdown()
ray.init(local_mode=True)

env = gym.make(
    'gym_cpr_grid:CPRGridEnv-v0', 
    n_agents=n_agents, 
    grid_width=grid_width, 
    grid_height=grid_height,
    initial_resource_probability=0.2,
    #fov_squares_front=21,
    global_obs=True
)

experiment_analysis = rllib.rllib_baseline(
    "dqn",
    n_agents,
    grid_width,
    grid_height,
    wandb_project,
    wandb_api_key,
    rllib_log_dir,
    max_episodes,
    tagging_ability=tagging_ability,
    gifting_mechanism=gifting_mechanism,
    num_workers=num_workers,
    jupyter=False,
    seed=seed
)