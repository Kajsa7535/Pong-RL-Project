import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float() 
    elif 'ALE/Pong-v5' in env:  
        obs = torch.tensor(obs, device=device).float() / 255.0       
        return obs
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
