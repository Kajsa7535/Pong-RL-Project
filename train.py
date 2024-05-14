import argparse
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

NUM_RUN = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--target_update_frequency', type=int, required=False, help='Target update frequency for DQN')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]
    #env_config['target_update_frequency'] = args.target_update_frequency

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device) # duplicate network for target network
    target_dqn.load_state_dict(dqn.state_dict()) # Initialize target network with the same weights as the DQN.

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    results = {
        "episodes": [],
        "returns": [],
        "memory_size": env_config['memory_size'],
        "n_episodes": env_config['n_episodes'],
        "batch_size": env_config['batch_size'],
        "target_update_frequency": env_config['target_update_frequency'],
        "train_frequency": env_config['train_frequency'],
        "gamma": env_config['gamma'],
        "lr": env_config['lr'],
        "eps_start": env_config['eps_start'],
        "eps_end": env_config['eps_end'],
        "anneal_length": env_config['anneal_length'],
        "n_actions": env_config['n_actions']
    }
    best_mean_return = -float("Inf")
    step = 0
    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()
        
        while not terminated:
            step += 1
            dqn.current_step += 1
            obs = preprocess(obs, env=args.env).unsqueeze(0)
            action = dqn.act(obs)
            action_item = action.item()
            next_obs, reward, terminated, truncated, info = env.step(action_item) 

            torch_obs = torch.tensor(obs)
            obs = next_obs

            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            else:
                next_obs = None

            if next_obs is not None:
                torch_next_obs = torch.tensor(next_obs, device=device)
            else:
                torch_next_obs = None
            torch_action = torch.tensor(action)
            torch_reward = torch.tensor(reward)
            memory.push(torch_obs, torch_action, torch_next_obs, torch_reward)

            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            if step % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            results['episodes'].append(episode+1)
            results['returns'].append(mean_return)

            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best_{args.target_update_frequency}.pt')
    
    # Save the results to a file
    with open(f'scores/results_target_{NUM_RUN}.json', 'w') as f:
        json.dump(results, f)
        
    # Close environment after training is completed
    env.close()