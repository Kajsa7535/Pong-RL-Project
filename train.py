import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)

    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    # TODO: Create and initialize target Q-network.

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])
    print(memory)

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    batch_size = env_config['batch_size']
    #batch_obs = np.zeros((batch_size,4))

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()
        print(f'before {obs}')
        obs = preprocess(obs, env=args.env).unsqueeze(0)
        print(f'after {obs}')
        steps = 0
        while not terminated:
            # TODO: Get action from DQN.
            obs = obs.cpu()  # Assuming obs is a PyTorch tensor and it's on CPU
            obs = obs.repeat(batch_size, 1)

            action = dqn.act(obs.to(device))
            print("Shape:", action.shape)


            # Act in the true environment.
            for i in range(batch_size):
                next_obs, reward, terminated, truncated, info = env.step(action[i].item())
                if terminated:
                    next_obs = None
                    break
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                memory.push(obs[i], action[i], reward, next_obs)

            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(obs, env=args.env).unsqueeze(0)
            else:
                next_obs = None
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            
            
            obs = next_obs if next_obs is not None else obs

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            if steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            obs = next_obs
            # TODO: Update the target network every env_config["target_update_frequency"] steps.

    
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
