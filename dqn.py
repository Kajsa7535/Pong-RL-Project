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

         # Store None as it is, or convert to tensor as needed
        #self.memory[self.position] = (obs, action, next_obs if next_obs is None else torch.tensor(next_obs, device=device), reward)
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
        self.current_step = 0

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
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
        """Selects an action with an epsilon-greedy exploration strategy."""
        # DONE: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.

        # DONE: Implement epsilon-greedy exploration.1 TODO: Är detta verkligen så man gör???


       
        random_number = random.uniform(0, 1)
        epsilon = self.epsilon()
    
        if exploit or random_number > epsilon:
            if exploit:
                with torch.no_grad():
                    actions =  torch.argmax(self.forward(observation), dim=1).unsqueeze(1)
                
            else:
 
                actions =  torch.argmax(self.forward(observation), dim=1).unsqueeze(1)
        else:
            actions =  torch.randint(0, self.n_actions, (observation.size(0), 1), device=device)

       
        return actions

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transitions = memory.sample(dqn.batch_size)
    states, actions, next_states, rewards = transitions
    # Iterate over the elements of next_state
    # Get the shape of a non-None tensor in next_state
    non_final_mask = torch.tensor([s is not None for s in next_states], device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in next_states if s is not None])
    non_final_next_states = non_final_next_states.squeeze(1).to(device)

    next_state_with_zeros = tuple(tensor if tensor is not None else torch.zeros(1, 4) for tensor in next_states)
    next_states = torch.stack(next_state_with_zeros).to(device)
    states = torch.stack(states).to(device).squeeze(1)  
    actions = torch.stack(actions).to(device).squeeze(1)  
    rewards = torch.stack(rewards).to(device)
    
    # Compute Q-values for current states and actions
    next_q_values = torch.zeros((dqn.batch_size, 1), device=device)
    current_q_values = dqn(states).gather(1, actions)
   
    next_q_values[non_final_mask] = target_dqn(non_final_next_states).max(-1)[0].unsqueeze(1).detach()


    # Compute the expected Q values (targets)
    expected_q_values = (next_q_values.squeeze(-1) * dqn.gamma) + rewards

    # Compute loss.
    loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(-1))

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    return loss.item()


