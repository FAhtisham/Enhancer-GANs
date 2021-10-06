#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:37:26 2020

@author: ahtisham
"""
import torch
from torch import optim
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import torch.autograd as autograd


import matplotlib.pyplot as plt
import numpy as np
import os.path
from src.gumbel import *
from src.parser import parameter_parser
from src.utils import *
from src.models import Discriminator
from src.models import Generator

class WEnhancerGAN:
    def __init__(self, args, num_chars=4):
        
        #self.__init_data__(args)
        
        # assign the parameters from args parser
        self.batch_size = args.batch_size
        self.hidden = args.hidden_dim
        self.lr = args.learning_rate
        self.epochs = args.epochs
        self.sequence_length = args.max_len
        self.discriminator_steps = args.discriminator_steps
        self.generator_steps = args.generator_steps
        self.directory = args.directory
        self.lam = args.lam
        self.num_chars = num_chars
        
        # call preprocessing class from utils
        self.preprocessing = Preprocessing(args)
        
        self.device = "cuda:2"
        
        
            
    def __init_data__(self, args):
        
        # function used from the utils files (see utils for details)
        self.preprocessing = Preprocessing(args)
        
        # read fasta of positive sequences (enhancers)
        self.preprocessing.load_data()
        
        if (os.path.exists("oneHotEncodedData.npy")):
            #self.data = np.load("oneHotEncodedData.npy")
            print("data p")
            #print("One hot encoded data present !!! \nShape :", self.data.shape)
        else:
            print("Reading and One hot encoding the sequences")
            self.preprocessing.sequencesToOneHotEncoding()
        

        
        # read and convert the origina
        
        
        '''
        # preparing train and test data
        self.preprocessing.prepare_train_test()
        self.preprocessing.prepare_tokens()
        
        raw_x_train = self.preprocessing.x_train
        raw_x_test = self.preprocessing.x_test
        
        
        self.y_train = self.preprocessing.y_train
        self.y_test = self.preprocessing.y_test
          
        '''
        
    def build_GAN_model(self):
        
        # defining the models
        print(self.num_chars, self.sequence_length, self.batch_size, self.hidden)
        self.Generator = Generator(self.num_chars, self.sequence_length, self.batch_size, self.hidden).to(self.device)
        self.Discriminator = Discriminator(self.num_chars, self.sequence_length, self.batch_size, self.hidden).to(self.device)

        # defining the optimizers 
        self.d_optim = optim.Adam(self.Discriminator.parameters(), lr= self.lr, betas = (0.5, 0.9))
        self.g_optim = optim.Adam(self.Generator.parameters(), lr= self.lr, betas = (0.5, 0.9))

        print("Models have been built...")
        
    
        
        
args = parameter_parser()	
wgan = WEnhancerGAN(args)
wgan.build_GAN_model()

#wgan.data[0]
        
        
        
        
        
        
        
        
        

