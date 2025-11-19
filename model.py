# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from features import INPUT_SIZE, ACTION_SPACE_SIZE

class PVModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Common Body
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)

        # Policy Head (Move probabilities)
        self.policy_head = nn.Linear(128, ACTION_SPACE_SIZE)

        # Value Head (Win probability: -1 to 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)) # Tanh for range [-1, 1]
        
        return policy_logits, value

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
