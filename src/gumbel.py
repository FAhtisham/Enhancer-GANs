#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:40:10 2020

@author: ahtisham
"""
""" https://github.com/ericjang/gumbel-softmax/    (MIT license)"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def sample_gumble(shape, epsilon = 1e-20, output=None):
    if output != None:
        U = output.resize_(shape).uniform()
    else:
        U = torch.rand(shape)
    
    return - torch.log(epsilon - torch.log(U + epsilon))


def gumble_softmax_sample(logits, epsilon = 1e-20, temp = 1 ):
    dimensions = logits.dim()
    gumbel_noise = sample_gumble(shape = logits.size(), epsilon = epsilon, output = logits.data.new() )
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / temp, dimensions - 1)

def gumbel_softmax(logits, temperature = 1, hard = False, epsilon = 1e-20):
    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumble_softmax_sample(logits, temp = temperature, epsilon = epsilon)
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
        

        


    
    
    