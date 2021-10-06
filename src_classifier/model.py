# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:04:26 2020

@author: Ahtisham The greatest professor
"""

import torch
import torch.nn as nn

class EnhancerClassifier(nn.ModuleList):
    def __init__(self, args):
        super(EnhancerClassifier, self).__init__()
        
        # Hyperparameters
        self.batch_size = args.batch_size
        self.hidden_dim= args.hidden_dim
        self.LSTM_layers= args.lstm_layers
        self.vocab_size= args.vocab_size
        
        # Droput layer
        self.dropout = nn.Dropout(0.2)
        
        # Embedding layer to get a lookup table
        self.embedding = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.hidden_dim, padding_idx=0)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features= self.hidden_dim*2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(in_features= self.hidden_dim*2, out_features=1)
    
    def forward(self, x):
        
        # defining the hidden and cell state of rnn cells
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to("cuda:3")
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to("cuda:3")
        
        # initialization for the hidden and current cell state
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        
        # Sequence passed through the embedding layer
        out = self.embedding(x)
        
        # Pass it to LSTMs
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = self.dropout(out)
        
        # last hidden state 
        out = torch.relu_(self.fc1(out[:,-1,:]))
        
        # Pass it to dropout
        out = self.dropout(out)
        
        # Pass the output of second fully connected layer to sigmoid 
        out = torch.sigmoid(self.fc2(out))
        
        return out
    
    
    '''
    def init_hidden(self, x):
        
        # in terms of rnn we need to define the hidden and cell state definition\
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        
        # initialization to the hidden and cells states
        return torch.nn.init.xavier_normal_(h), torch.nn.init.xavier_normal_(c)
        
        '''
        
        
        
        