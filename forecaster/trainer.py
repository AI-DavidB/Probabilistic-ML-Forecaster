import torch
import torch.optim as optim
import numpy as np

class ForecasterTrainer:
    """
    Handles the training loop using Negative Log-Likelihood (NLL) loss.
    """
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def nll_loss(self, mu, sigma, target):
        """
        Gaussian Negative Log-Likelihood Loss.
        """
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(target).mean()

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        mu, sigma = self.model(x)
        loss = self.nll_loss(mu, sigma, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
