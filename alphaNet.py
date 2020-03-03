import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
"The input to the neural network is an N x N x (MT + L) image stack that represents state
using a concatenation of T sets of M planes of size N x N." (AlphaZero 14)
see encode_board
"""
class ConvBlock(nn.Module):
    """
    "The convolutional block applies the following modules:
    (1) A convolution of 256 filters of kernel size 3×3 with stride 1
    (2) Batch normalization
    (3) A rectifier nonlinearity" (AlphaGo supplement)
    256 filters has been changed to a lower number simply because tictactoe is way simpler than Go"
    """
    def __init__(self, in_filters=3, out_filters=32):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        # relu
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class ResBlock(nn.Module):
    """
    "Each residual block applies the following modules sequentially to its input:
    (1) A convolution of 256 filters of kernel size 3×3 with stride 1
    (2) Batch normalization
    (3) A rectifier nonlinearity
    (4) A convolution of 256 filters of kernel size 3×3 with stride 1
    (5) Batch normalization
    (6) A skip connection that adds the input to the block
    (7) A rectifier nonlinearity" (AlphaGo supplement)
    """
    def __init__(self, in_filters=32, out_filters=32):
        super(ResBlock,self).__init__()
        # store res
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_filters)
        # relu
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_filters)
        # add res
        # relu
    
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x += res
        x = F.relu(x)
        return x

class PolicyHead(nn.Module):
    """
    The output of the residual tower is passed into two separate ‘heads’ for
    computing the policy and value. The policy head applies the following modules:
    (1) A convolution of 2 filters of kernel size 1×1 with stride 1
    (2) Batch normalization
    (3) A rectifier nonlinearity
    (4) A fully connected linear layer that outputs a vector of size 19^2+ 1= 362, 
    corresponding to logit probabilities for all [moves]" (AlphaGo supplement)
    in this case, pass is not a legal move, and the board is 3x3, so the output layer is size 9
    """
    def __init__(self, action_space=9, board_size=9, in_filters=32):
        super(PolicyHead, self).__init__()
        self.board_size = board_size # 1d "length" of board size
        
        self.conv1 = nn.Conv2d(in_filters, 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2)
        # relu
        self.fc1 = nn.Linear(self.board_size*2, action_space)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, self.board_size*2)
        x = self.softmax(self.fc1(x))
        return x

class ValueHead(nn.Module):
    """
    "The value head applies the following modules:
    (1) A convolution of 1 filter of kernel size 1×1 with stride 1
    (2) Batch normalization
    (3) A rectifier nonlinearity
    (4) A fully connected linear layer to a hidden layer of size 256
    (5) A rectifier nonlinearity
    (6) A fully connected linear layer to a scalar
    (7) A tanh nonlinearity outputting a scalar in the range [−1, 1]" (AlphaGo supplement)
    Again, 256 is kind of excessive, so the number was brought down
    """
    def __init__(self, board_size=9, in_filters=32, hidden_dims=32):
        super(ValueHead, self).__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(in_filters, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        # relu
        self.fc1 = nn.Linear(self.board_size*1, hidden_dims)
        self.dropout1 = nn.Dropout(.3)
        # relu
        self.fc2 = nn.Linear(hidden_dims, 1)
        self.dropout2 = nn.Dropout(.3)
        # tanh
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, self.board_size*1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(torch.tanh(self.fc2(x)))
        return x

class AlphaNet(nn.Module):
    """
    "The input features s_t are processed by a residual tower that consists of a single
    convolutional block followed by either 19 or 39 residual blocks...
    The output of the residual tower is passed into two separate ‘heads’ for
    computing the policy and value." (AlphaGo supplement)
    Again, probably don't need 19 res blocks for tictactoe. Will experiment
    """
    def __init__(self, board_layers=0, board_size=0, action_space=0, n_filters=0, n_hidden=0, n_res=0):
        super(AlphaNet, self).__init__()
        self.conv_block1 = ConvBlock(in_filters=board_layers, out_filters=n_filters)
        
        self.n_res = n_res
        for k in range(n_res):
            setattr(self, f"res_block{k+1}", ResBlock(in_filters=n_filters, out_filters=n_filters))
        
        self.policy_head = PolicyHead(action_space=action_space, board_size=board_size, in_filters=n_filters)
        
        self.value_head = ValueHead(board_size=board_size, in_filters=n_filters, hidden_dims=n_hidden)
    
    def forward(self, x):
        x = self.conv_block1(x)
        for k in range(self.n_res):
            x = getattr(self, f"res_block{k+1}")(x)
        p = self.policy_head(x)
        v = self.value_head(x).squeeze(-1)
        return p,v
    
    def load_weights(self, weights_filepath):
        self.load_state_dict(torch.load(weights_filepath))

class NoResAlphaNet(nn.Module):
    def __init__(self, board_layers=0, board_size=0, action_space=0, n_filters=0, n_hidden=0, n_conv=0):
        super(NoResAlphaNet, self).__init__()
        
        self.n_conv = n_conv
        for k in range(self.n_conv):
            setattr(self, f"conv_block{k+1}", ConvBlock(in_filters=(board_layers if k==0 else n_filters), out_filters=n_filters))

        self.conv_out = n_filters*9 # TIC TAC TOE ONLY
        self.fc1 = nn.Linear(self.conv_out, n_hidden)
        self.dropout1 = nn.Dropout(.3)
        # relu

        self.fc2 = nn.Linear(n_hidden, n_hidden//2)
        self.dropout2 = nn.Dropout(.3)
        # relu

        self.policy_head = nn.Linear(n_hidden//2, action_space)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.value_head = nn.Linear(n_hidden//2, 1)
        # tanh
    
    def forward(self, x):
        for k in range(self.n_conv):
            x = getattr(self, f"conv_block{k+1}")(x)

        x = x.view(-1, self.conv_out)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))

        p = self.policy_head(x)
        p = self.softmax(p)

        v = self.value_head(x).squeeze(-1)
        v = torch.tanh(v)
        return p,v
    
    def load_weights(self, weights_filepath):
        self.load_state_dict(torch.load(weights_filepath))
        

class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    """
    "Specifically, the parameters θ are adjusted by gradient 
    descent on a loss function l that sums over the mean­-squared error and 
    cross-­entropy losses, respectively:
    (p,v)=f_θ(s) and l=(z−v)^2 − π^Tlog(p)+c||θ||^2     (1)" (AlphaGo Zero 355)
    """
    def forward(self, pi, p, z, v):
        v_err = (z - v)**2 # mean squared error
        p_err = torch.sum(pi*p, dim=-1)
        loss = (v_err.view(-1) - p_err).mean()
        # missing the weight regularization, maybe not needed for t3
        return loss