import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.value_head(hidden_states[:, -1, :])
