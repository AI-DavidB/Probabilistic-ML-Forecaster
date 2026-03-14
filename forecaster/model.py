import torch
import torch.nn as nn

class ProbabilisticLSTM(nn.Module):
    """
    LSTM-based architecture that outputs a Gaussian Distribution (Mean and StdDev).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(ProbabilisticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Dual heads: one for Mean, one for Standard Deviation (using Softplus for positivity)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        
        mu = self.mean_head(last_step)
        # Sigma must be positive; add a small epsilon for numerical stability
        sigma = self.softplus(self.sigma_head(last_step)) + 1e-6
        
        return mu, sigma
