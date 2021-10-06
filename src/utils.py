# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:06:15 2020

@author: Ahtisham
"""

import numpy as np


from tqdm import tqdm

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
        print(len(self.longer_sequences))
            
    def write_long_seq_file(self):
        f = open("long_enhancer.FASTA", "a")
        for line in self.longer_sequences:
            f.write(line)
        f.close()

    # Combining the shorter and longer sequences (positive ones only)               
    def combine(self):
        self.sequences =  self.shorter_sequences +self.longer_sequences

        
    
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

        # pad short seqs
        self.short_to_augment()
            
        # Subsampling from longer sequences
        self.long_to_augmentation()

        # Combining the long and short sequences
        self.combine()

        print("Longer positive sequences=", len(self.longer_sequences), "Shorter positve sequences=", len(self.shorter_sequences))
        print("Total no of  sequences=", len(self.sequences))

    def baseToOnehotEncoding(self, base):
        if base == 'A':
            return np.array([1,0,0,0,0])
        elif base == 'G':
            return np.array([0,1,0,0,0])
        elif base == 'C':
            return np.array([0,0,1,0,0])
        elif base == 'T':
            return np.array([0,0,0,1,0])
        elif base == 'P':
            return np.array([0,0,0,0,1])

    def sequencesToOneHotEncoding(self):
        self.all_samples = np.zeros((len(self.sequences), self.max_len, 5))
        current_sample = np.zeros((self.max_len,5))
        sample_count = 0

        for line in tqdm(self.sequences):
            line = line.upper()
            first_base = True

            base_count = 0
            for base in line:

                if first_base == True:
                    first_base = False
                    current_sample = np.zeros((self.max_len,5))
                    current_sample[base_count] = self.baseToOnehotEncoding(base)
                    base_count += 1
                else:
                    current_sample[base_count] = self.baseToOnehotEncoding(base)
                    base_count += 1

            self.all_samples[sample_count] = current_sample
            sample_count += 1
        print("All Sample Size: ", self.all_samples.shape)
        np.save("oneHotEncodedData.npy", self.all_samples)
        print("one Hot Encoding  Done.....")
        
        
    # add padding to short sequences
    def short_to_augment(self):
        for i in range(len(self.shorter_sequences)):
            if (len(self.shorter_sequences[i])<300):
                temp = self.shorter_sequences[i]
                while (len(temp) <300):
                    temp = temp + "P"
                self.shorter_sequences[i] = temp
        
        
    def single_sequenceToOneHotEncoding(self, sequence):
        current_sample = np.zeros((len(sequence),5))
        if len(sequence)<300:
            for i in range(300 - len(sequence)): sequence +=  "P"
        sample_count = 0
        
        first_base = True

        base_count = 0
        for base in sequence:
            if first_base == True:
                first_base = False
                current_sample = np.zeros((self.max_len,5))
                current_sample[base_count] = self.baseToOnehotEncoding(base)
                base_count += 1
            else:
                current_sample[base_count] = self.baseToOnehotEncoding(base)
                base_count += 1
        return current_sample

            