#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:56:09 2020

@author: ahtisham
"""

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

#import .gumbel
#from gumbel.gumbel import gumbel_softmax
'''
# create resnet blocks
class ResnetBlock(nn.Module):
    def __init__(self, hidden):
        self.hidden = int(hidden)
        

        super(ResnetBlock, self).__init__()
        
        # first conv layer
        self.conv1 = nn.Conv1d(in_channels = self.hidden, out_channels = self.hidden, kernel_size= 5, padding=2)
        
        # second conv layer
        self.conv2 = nn.Conv1d(in_channels = self.hidden, out_channels =  self.hidden, kernel_size = 5, padding=2)
        
    def forward(self, input):
        
        # apply relu
        output = torch.relu_(input)
        
        # apply first conv
        output = self.conv1(output)
        
        # apply relu 
        output = torch.relu_(output)
        
        # apply second conv
        output = self.conv2(output)
        
        return input + (0.3*output)


# generator class
class Generator(nn.Module):
    def __init__(self, num_chars, sequence_length, batch_size, hidden):
        super(Generator, self).__init__()
        
        # assign parameters
        self.num_chars = num_chars
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden = hidden
        
        # apply conv layer
        self.conv1 = nn.Conv1d(self.hidden, self.num_chars, kernel_size=1)
        
        # apply fully connected layer
        self.fc1 = nn.Linear(128, self.hidden*self.sequence_length)
        
        # apply resnet blocks
        self.resnet_blocks = nn.Sequential(
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
        )


    def forward(self, latent_vector):
        
        # apply full connected layer
        output = self.fc1(latent_vector)
        
        # reshaping
        output = output.view(-1, self.hidden, self.sequence_length) 
        
        # apply res blocks
        output = self.resnet_blocks(output)
        
        # apply conv layer
        output = self.conv1(output)
        print(output)
        # apply transpose
        output = output.transpose(1, 2)
        shape = output.size()

        # apply contiguos 
        output = output.contiguous()

        # reshape
        output = output.view(self.batch_size*self.sequence_length, -1)
        #print("before",output.size())

        # apply gumbel
        output = gumbel_softmax(output, 0.5)

        
        # reshape and return
        return output.view(shape) # (BATCH_SIZE, sequence_length, len(charmap))

class Discriminator(nn.Module):
    def __init__(self, num_chars, sequence_length, batch_size, hidden):
        super(Discriminator, self).__init__()
        
        # assign parameters
        self.num_chars = num_chars
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden = hidden
        
        
        self.resnet_blocks = nn.Sequential(
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
            ResnetBlock(self.hidden),
        )
        
        # first conv layer
        self.conv1d = nn.Conv1d(self.num_chars, self.hidden, 1)
        
        # first fully connected layer
        self.fc1 = nn.Linear(self.sequence_length*self.hidden, 1)

    def forward(self, input):

        #print("step 1: ",input.size())
        # transpose the input
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), sequence_length)

        #print("step 2: ", output.size())
        # apply first conv
        output = self.conv1d(output)
        
        # apply resent blocks
        output = self.resnet_blocks(output)
        
        # reshape
        output = output.view(-1, self.sequence_length*self.hidden)
        
        # applye the fully connected layer
        output = self.fc1(output)
        
        return output

'''



import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
#from utils.torch_utils import *

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        #print(input)
        output = self.res_block(input)
        #print(output)
        return input + (0.5 * output)

class Generator(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        #print("before", noise)
        output = self.fc1(noise)
        #print("after",output)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Discriminator(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Discriminator, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        return output

###############################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:40:10 2020

@author: ahtisham
"""
""" https://github.com/ericjang/gumbel-softmax/    (MIT license)"""



def sample_gumble(shape, epsilon = 1e-20, output=None):
    if output != None:
        U = output.resize_(shape).uniform_()
    else:
        U = torch.rand(shape)
    return - torch.log(epsilon - torch.log(U + epsilon))


def gumble_softmax_sample(logits, epsilon = 1e-20, temp = 1 ):
    dimensions = logits.dim()
    gumbel_noise = sample_gumble(shape = logits.size(), epsilon = epsilon, output = logits.data.new() )
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / temp, dimensions - 1)

def gumbel_softmax(logits, temp=1, hard=False, eps=1e-20):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temp: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints: - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumble_softmax_sample(logits, epsilon=1e-20, temp=temp)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y
        

'''
    import torch
    import torch.nn as nn
    x = torch.randn(128, 50000, device='cuda')
    fc1 = nn.Linear(50000, 512).cuda()
    y = fc1(x)
'''