import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'], default='CartPole-v0')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
   

    env_config = ENV_CONFIGS[args.env]
    

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device) #duplicate network for target network
    target_dqn.load_state_dict(dqn.state_dict()) # Initialize target network with the same weights as the DQN.
    # DONE: Create and initialize target Q-network.
    

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    step = 0
    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()
        
        while not terminated:
            step+=1
            dqn.current_step += 1
            # DONE: Get action from DQN.
            obs = preprocess(obs, env=args.env).unsqueeze(0)

            action = dqn.act(obs)

            # Act in the true environment.
            action_item = action.item() #TODO WILL THIS WORK WITH LARGER BATCH SIZE?
            next_obs, reward, terminated, truncated, info = env.step(action_item) 

            torch_obs = torch.tensor(obs)
            obs = next_obs

            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            else:
                next_obs = None
            # DONE: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            
            if next_obs is not None:
                torch_next_obs = torch.tensor(next_obs, device=device)
            else:
                torch_next_obs = None  # Set to None if there's no next observation
            torch_action = torch.tensor(action)
            torch_reward = torch.tensor(reward)
            memory.push(torch_obs, torch_action, torch_next_obs, torch_reward)

            # DONE: Run DQN.optimize() every env_config["train_frequency"] steps.
            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # DONE: Update the target network every env_config["target_update_frequency"] steps.
            if step % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
