import torch
from torch import nn

class NetworkCNN(nn.Module):
    def __init__(self, feature_number, num_currencies, window_size):
        '''
        Args:
            feature_number = feature counts (high, low, open)
            num_currencies =  number of currencies
            window_size = historic time period
        '''
        super(NetworkCNN, self).__init__()
        self.feature_number = feature_number
        self.n_coins = num_currencies
        self.w_size = window_size
        self.relu = nn.ReLU()
        self.conv1 = torch.nn.Conv2d (self.feature_number , 2, (1,3), bias = False)
        self.conv2 = torch.nn.Conv2d ( 2, 20, (1,48), bias = False)
        self.conv3 = torch.nn.Conv2d ( 21, 1, (1,1), bias = False)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x, W):
        
        #input_tensor =[batch_size, feature_number, num_currencies, window_size] 
        # x = x.permute(0,2,3,1) #[batch, num_currencies, window, features]
        input_dim = x.shape[0]
        assert W.shape[1]==self.n_coins, "Number of currencies are not matching"
        
        # print(f"============== input size -> {x.size()} ,  portfolio vector size ->   {W.shape}")

        # import pdb; pdb.set_trace()
        x= self.conv1(x)
        x = self.relu(x)
        
        # print(f"============== conv 1x3 ->  {x.shape}")
        x =  self.conv2(x)
        x = self.relu(x) 
        
        # print(f"============== conv 1x48 ->   {x.shape}") # [batch_size, 20, num_currencies, 1]
        x= torch.concat((x, W.unsqueeze(1)), dim=1) ## concat previous weights with the conv features
        # import pdb; pdb.set_trace()
        x =  self.conv3(x)
        
        
        # print(f"============== conv 1x1 ->   {x.shape}")
        bias_cash = torch.ones((x.shape[0], 1, 1, 1))  
        
        # print(f"============== bias ->   {bias_cash.shape}")
        x= torch.concat(( bias_cash, x), dim=2)  
        
        out = self.softmax(x).squeeze(1)
        
        # print(f"============== output ->  {out.shape}")
        return out