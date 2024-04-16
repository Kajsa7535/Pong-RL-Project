import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        if(x == None):
            return 0 #return 0 for terminating states
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def epsilon(self):
            """Returns the current epsilon value based on annealing schedule."""
            if self.current_step >= self.anneal_length:
                return self.eps_end
            else:
                eps = max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * self.current_step / self.anneal_length)
                return eps

    def act(self, observation, exploit=False):
            # TODO: Implement action selection using the Deep Q-network. This function
            #       takes an observation tensor and should return a tensor of actions.
            #       For example, if the state dimension is 4 and the batch size is 32,
            #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
            # TODO: Implement epsilon-greedy exploration.
            """Selects an action with an epsilon-greedy exploration strategy."""
            epsilon = self.epsilon()
            if exploit or random.random() > epsilon:
                with torch.no_grad():
                    return torch.argmax(self.forward(observation), dim=1).unsqueeze(1)
            else:
                return torch.randint(0, self.n_actions, (observation.size(0), 1))
    

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and 
    samples = memory.sample(dqn.batch_size)
    sample_batch = zip(*samples)
    
    #concatenate so that there are
    #four tensors in total: observations, actions, next observations and rewards.
    observations_tensor = torch.cat(sample_batch.obs)
    action_tensor = torch.cat(sample_batch.action)
    next_obs_tensor = torch.cat(sample_batch.next_obs) # not getting the terminating states
   # next_obs_tensor = torch.cat([x for x in sample_batch.next_obs if x is not None]) # not getting the terminating states
    reward_tensor = torch.cat(sample_batch.reward)

    #  Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observations_tensor.to(device)
    action_tensor.to(device)
    next_obs_tensor.to(device)
    reward_tensor.to(device)

     
    # Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = dqn(observations_tensor).gather(1, action_tensor)
    
   
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    # if terminating : only reward 
    # else r + gamme * max q value
    #add reward and gamma


    q_value_targets = target_dqn(next_obs_tensor).gather(1, action_tensor)
    q_value_targets = reward_tensor + dqn.gamma * q_value_targets.max(1)[0].unsqueeze()


    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()

