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
from gymnasium.wrappers import AtariPreprocessing

torch.backends.cudnn.enabled = False

NUM_RUN = "second_run_atari"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"CUDA is available. GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=10, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--target_update_frequency', type=int, required=False, help='Target update frequency for DQN')
parser.add_argument('--load-checkpoint', type=str, help='Path to stored DQN model.')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]
    #env_config['target_update_frequency'] = args.target_update_frequency

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    if args.load_checkpoint:
        state_dict = torch.load(args.load_checkpoint)
        dqn.load_state_dict(state_dict)
    target_dqn = DQN(env_config=env_config).to(device)  # duplicate network for target network
    target_dqn.load_state_dict(dqn.state_dict()) # Initialize target network with the same weights as the DQN.

    # Create replay memory.
    memory = ReplayMemory(env_config['replay_memory_capacity'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    results = {
        "episodes": [],
        "returns": [],
        "observation_stack_size": env_config['observation_stack_size'],
        "memory_size": env_config['replay_memory_capacity'],
        "batch_size": env_config['batch_size'],
        "target_update_frequency": env_config['target_update_frequency'],
        "train_frequency": env_config['train_frequency'],
        "gamma": env_config['gamma'],
        "lr": env_config['lr'],
        "eps_start": env_config['eps_start'],
        "eps_end": env_config['eps_end'],
        "anneal_length": env_config['anneal_length'],
        "n_actions": env_config['n_actions'],
        "observation_stack_size": env_config['observation_stack_size']
    }
    best_mean_return = -float("Inf")
    step = 0
    obs_stack_size = env_config['observation_stack_size']
    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0).to(device)
      

        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)
   
        
        while not terminated:
            step += 1
            dqn.current_step += 1
            action = dqn.act(obs_stack) 

            action_item = action.item() +2

            next_obs, reward, terminated, truncated, info = env.step(action_item)
            #make next obs to tensor

            next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(0)), dim=1).to(device)

            torch_obs_stack = obs_stack.clone().detach().to(device)
            torch_action = action.clone().detach().to(device)
            torch_reward = torch.tensor([reward], dtype=torch.float32).to(device)
            
            memory.push(torch_obs_stack, torch_action, next_obs_stack, torch_reward)
            
            obs_stack = next_obs_stack

            if step > env_config["learning_start"] and step % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            if step > env_config["learning_start"] and step % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            print(f'dqn.current_step: {dqn.current_step}, dqn.epsilon(): {dqn.epsilon()}')
            results['episodes'].append(episode+1)
            results['returns'].append(mean_return)

            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best_{args.target_update_frequency}.pt')
                torch.save(dqn.state_dict(), f'models/{args.env}_best_{args.target_update_frequency}_state_dict.pt')
    
    # Save the results to a file
    with open(f'scores/results_target_explore_{NUM_RUN}.json', 'w') as f:
        json.dump(results, f)
        
    # Close environment after training is completed
    env.close()