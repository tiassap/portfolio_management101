import torch
import numpy as np
import deque
from src.network import #...



class DPG(object):
    """
    Deterministic Policy Gradient
    """

    def __init__(self, config, dataset) -> None:
        self.config = config # yaml parser (can be accessed as dictionary). See `config/ ... .yml` file
        self.price_data = dataset # Dataset source is defined in `run.py`. Example = marketData_CSV()
        self.buffer = PVM() # Replay buffer
        self.NNmodel = None # should be replaced by class Neural_Network()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.beta = config["beta"] # = 5e-5, probability-decaying rate determining the shape of the probability distribution for sampling tb for training NN
        self.nb = config["window-size_nb"] # = 50, window size (size of X).
        self.Nb = config["mini-batch_Nb"] # = 50, mini batch size
        self.cs = config["cs"] # 0.0025 commision rate

        
    def train(self):
        """
        Run training process
        """
        t = 0
        while True:
            t += 1

            # Because X size is nb --> 50
            if t >= self.nb:
                # Get a batch price data at time step t, take portfolio vector from replay buffer, and do forward pass
                X = self.get_X(t)
                w = self.buffer.get_previous_w()
                w_out = self.take_action(X, w) # forward pass of neural network

            # Store sample path into replay buffer
            self.buffer.store_portfolio_vector(w_out)

            # Start training after ... time steps (after portvolio vector memory filled), and update neural network every ... time step freq
            if t >= train_start and t % freq_train == 0 :
            # Learning one step using batch of data from replay buffer.
                train_batch = self.get_sample_batch()
                self.update_step(train_batch)

            # Finish training at these conditions
            if t >= max_t_length or t >= dataset_end_period:
                break
    
    def get_X(self, t):
        """
        get X input at time step t
        """
        return self.price_data[:, :, t-self.nb:t]

    def calc_portValue(self, Y, w):
        """
        Calculate portfolio value at given step: sum of ( price of each asset times weight of each asset)
        """
        pass

    def take_action(self, x, w):
        """
        Get action (portfolio weight vector) by doing inference on neural network
        """
        return self.NNmodel(x, w) # NN model take price data input and previous portfolio weight

    def update_step(self, train_batch):
        """
        One neural network training update step.
        Use sample batch.
        """
        X, prev_w = train_batch
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

    def get_sample_batch(self, t, beta, nb):
        """
        Get a batch of path sample randomly from self.buffer (PVM() class)
        """
        tb_sample = distribution_tb(t, beta, nb)
        X_sample = self.getX(tb_sample)
        w_sample = self.buffer.get_previous_w(tb_sample)

        return X_sample, w_sample

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

    def get_previous_w(self):
        """
        take latest portfolio vector.
        """
        pass

    
