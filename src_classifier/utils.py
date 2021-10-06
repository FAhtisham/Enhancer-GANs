# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:06:15 2020

@author: Ahtisham
"""
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

############## Class for the preprocessing of the data #######################

class Preprocessing:
    def __init__(self, args):
        
        # Reading the args from the args parser
        self.data= args.data
        self.max_len= args.max_len
        self.vocab_size = args.vocab_size
        self.test_size= 0.3
        self.sequences = list()
        self.longer_sequences = list()
        self.shorter_sequences = list()
        self.negative_sequences = list()
        self.tot_n_seq = 57459
        self.flag = 0
    
    # Function to read FASTA lines from FASTA file 
    def read_fasta(self, fp):
        name, seq = None, []
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name: yield (name, ''.join(seq))
                name, seq = line, []
            else:
                seq.append(line)
        if name: yield (name, ''.join(seq))
        
    # Function to differentiate among long >300 and short enhancers <300    
    def short_long_sequence(self):
        for i in range(len(self.sequences)):
            if (len(self.sequences[i])>300):
                self.longer_sequences.append(self.sequences[i])
            elif (len(self.sequences[i])<=300):
                self.shorter_sequences.append(self.sequences[i])
                
    # Subsampling from the longer sequences      (if seq > 300 break it down into smaller sequences )      
    def long_to_augmentation(self):
        StrideValue = 2
        TrainingDataset = []
        for i in range (len(self.longer_sequences)):
            while len(self.longer_sequences[i]) > 300:
                TrainingDataset.append(self.longer_sequences[i][0:300])
                self.longer_sequences[i] = self.longer_sequences[i][StrideValue:]

        self.longer_sequences = TrainingDataset
            
        
    # Combining the shorter and longer sequences (positive ones only)               
    def combine(self):
        self.sequences = self.longer_sequences #+ self.shorter_sequences
    
    # FASTA file reading function     
    def load_data(self):
        
        # Reading FASTA file
        with open(self.data) as fp:
            for name, seq in self.read_fasta(fp):
                self.sequences.append(seq)
            print("Sequences Read Succesfully !!!!")
        print(len(self.sequences))
            
        # Calling the function to create long and short sequences
        self.short_long_sequence()
            
        # Subsampling from longer sequences
        self.long_to_augmentation()
        
        # Combining the long and short sequences
        self.combine()

        print("Longer psequences=", len(self.longer_sequences), "Shorter psequences=", len(self.shorter_sequences))
        print("Total no of positive sequences=", len(self.sequences))
    
    # Preparing train test data    
    def prepare_train_test(self):
        
        # combining the positive and negative seqs
        self.sequences = self.sequences + self.negative_sequences
        X= np.array(self.sequences)
        
        # definnition of the labels
        n= np.zeros(shape=(len(self.negative_sequences)))
        p= np.ones_like(n)
        Y= np.concatenate((p,n), axis=0)
        
        # split train test data
        self.x_train, self.x_test, self.y_train, self.y_test= train_test_split(X,Y, test_size= self.test_size, shuffle=True)
    
    # Addition of the spaces within the sequences to get aid for tokenizater library
    def add_white_spaces(self, seq):
        return (" ".join(seq))
    
    # Adding white spaces in train test data
    def train_test_whitespaces(self, temp):
        for i in range(len(temp)):
            temp[i]= (self.add_white_spaces(temp[i]))
        return temp
    
    # Reading the negative sequences
    def add_background_sequences(self):
        with open("jan") as fp:
            for name, seq in self.read_fasta(fp):
                self.negative_sequences.append(seq)
            print(" Negative Sequences Read Succesfully !!!!")
        print("Total number of background seqs before augmentation:",len(self.negative_sequences))
    
    # subsampling from the negative seqs and restricting the tot number for class balances (p==n)
    def optimize_negative_sequences(self):
        StrideValue = 2
        TrainingDataset = []
        for i in range(len(self.negative_sequences)):
            while (len(self.negative_sequences[i]) > 300 and (len(TrainingDataset)  < len(self.sequences))):
                TrainingDataset.append(self.negative_sequences[i][0:300])
                self.negative_sequences[i] = self.negative_sequences[i][StrideValue:]
        self.negative_sequences = TrainingDataset
        print("Total number of background seqs after augmentation:",len(self.negative_sequences))
    
    # preparing text token from tokenizer
    def prepare_tokens(self):
        
        # tokens for the train data
        temp = self.x_train.tolist()
        self.x_train = np.array(self.train_test_whitespaces(temp))            
        self.tokens= Tokenizer(num_words = self.vocab_size)
        self.tokens.fit_on_texts(self.x_train)
        
        # tokens for the test data
        temp = self.x_test.tolist()
        '''if (self.flag < 1):
            print(len(temp))
            self.flag += 1'''
        self.x_test = np.array(self.train_test_whitespaces(temp)) 
        self.tokens= Tokenizer(num_words = self.vocab_size)
        self.tokens.fit_on_texts(self.x_test)
        print("Seq tokenization done....")
        
    # preparing numneric tokens
    def sequence_to_token(self, x):
        sequences = self.tokens.texts_to_sequences(x)
        temp = sequence.pad_sequences(sequences, maxlen=self.max_len)
        if (self.flag < 2):
            print(x[0])
            print(sequences[0])
            self.flag +=2
        return temp
    
    