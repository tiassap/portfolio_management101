import torch
import numpy as np
import deque
from src.network import #...


class DPG(object):
    """Deterministic Policy Gradient"""

    def __init__(self, config, dataset) -> None:
        self.config = config # yaml parser (can be accessed as dictionary). See `config/ ... .yml` file
        self.price_data = dataset # Dataset source is defined in `run.py`
        self.buffer = PVM() # Replay buffer
        self.NNmodel = None # should be replaced by class Neural_Network()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.portVal_history = [] # list of portfolio value history

        
    def train(self):
        """
        Run training process
        """
        episode = 0
        while True:
            episode += 1

            # Get a batch of random price data and take portfolio vector from replay buffer
            X = self.get_sample_batch()
            w = self.buffer.get_vector()
            w_out = self.take_action(X, w)

            # Store sample path into replay buffer
            self.buffer.store_portfolio_vector(w_out)

            # Learning one step using batch of data from replay buffer.
            self.update_step()

            # Add history of portfolio value
            self.portVal_history.append(self.calc_portValue())

            if episode >= max_episode_length or t >= dataset_end_period:
                break


    def calc_portValue(self, Y, w):
        """
        Calculate portfolio value at given step
        """
        pass

    def take_action(self, x, w):
        """
        Get action (portfolio weight vector) by doing inference on neural network
        """
        return self.NNmodel(x, w) # NN model take price data input and previous portfolio weight

    def update_step(self):
        """
        One neural network training update step.
        Use sample batch.
        """
        self.optimizer.zero_grad()
        loss = self.calc_loss()
        loss.backward()
        self.optimizer.step()

    def calc_loss(self):
        """
        Calculate loss function. 
        Use batch (take from self.get_batch())
        """
        pass

    def get_sample_batch(self, t):
        """
        Get a batch of path sample randomly from self.buffer (PVM() class)
        """
        self.buffer.get_batch_data()
        pass

    def save_model(self):
        """
        Save neural network parameter weights as `model.pth` and output rewards as Numpy .npy file.
        """
        pass

    def plot_output(self):
        """
        Create output plot.
        """
        pass

    def run_training(self):
        self.train() # Run training process. Arguments for self.train() will be defined here.


class PVM():
    """Portfolio Vector Memory; a stack of portfolio vectors in chronological order"""

    def __init__(self) -> None:
        # On initialization, add portfolio vector [1, 0, ..., 0] (1 for 'cash' currency)
        pass

    def store_portfolio_vector(self):
        """
        Store portfolio vector
        """
        pass

    def get_vector(self):
        """
        take latest portfolio vector.
        """
        pass

    
