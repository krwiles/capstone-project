import torch
import torch.nn as nn

class LSTMHybrid(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers    
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (hidden_states, cell_states)) 
        out = out[:, -1, :] 
        
        out = self.fc_layers(out)
        
        return out