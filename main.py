#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:37:26 2020

@author: ahtisham
"""
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import autograd
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable, grad

import matplotlib.pyplot as plt
import numpy as np
import os.path
#from src.gumbel import *
from src.parser import parameter_parser
from src.utils import *
from src.models import Discriminator
from src.models import Generator

from enhancer_classifer import EnhancerClassifier
from classifier_parser import parameter_parser2
import matplotlib.pyplot as plt

from tqdm import tqdm 

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class WEnhancerGAN:
    def __init__(self, args, num_chars=5):

        # function to retrieve dataset (augmented enhancers)
        self.__init_data__(args)

        # assign the parameters from args parser
        self.batch_size = args.batch_size
        self.hidden = args.hidden_dim
        self.lr_d = args.learning_rate_d
        self.lr_g = args.learning_rate_g
        self.epochs = args.epochs
        self.sequence_length = args.max_len
        self.discriminator_steps = args.discriminator_steps
        self.generator_steps = args.generator_steps
        self.directory = args.directory
        self.lam = args.lam
        self.num_chars = num_chars
        self.gpweight = 10
        self.real_gen_enhancers = list()
        self.replace_index = 0
        
        # set up the gpu device
        self.device = "cuda:3"
        
        # attributes for predictions and replacing values
        self.predictions = []
        self.args_c = parameter_parser2()
        self.t = EnhancerClassifier(self.args_c).to(self.device)
        self.t.load_state_dict(torch.load("model"))
        self.t.eval()

        # call preprocessing class from utils
        self.preprocessing = Preprocessing(args)



        # build the gan model
        self.build_GAN_model()

        self.use_cuda = True #if torch.cuda.is_available() else False

    def __init_data__(self, args):

        # function used from the utils files (see utils for details)
        self.preprocessing = Preprocessing(args)

        # read fasta of positive sequences (enhancers)
        self.preprocessing.load_data()
        self.jan_seq = self.preprocessing.longer_sequences

        # load if data exists
        if (os.path.exists("oneHotEncodedData.npy")):
            self.data = np.load("oneHotEncodedData.npy")
            #self.data = self.data[:1000] 
            print("One hot encoded data present !!! \nShape :" ,self.data.shape)
        else:

            # create data if it doesnt exist
            print("Reading and One hot encoding the sequences")
            self.preprocessing.sequencesToOneHotEncoding()
            print("Shape of Read Data:",self.data.shape)

     # function to build GAN
    def build_GAN_model(self):

        # defining the models
        self.Generator = Generator(self.num_chars, self.sequence_length, self.batch_size, self.hidden).to(self.device)
        self.Discriminator = Discriminator(self.num_chars, self.sequence_length, self.batch_size, self.hidden).to(self.device)

        # defining the optimizers
        self.d_optim = optim.Adam(self.Discriminator.parameters(), lr=self.lr_d, betas=(0., 0.9))
        self.g_optim = optim.Adam(self.Generator.parameters(), lr=self.lr_g, betas=(0., 0.9))

        print("Models have been built...")

    # calculate gradient penalty for WGAN GP
    def Gradient_Norm(self,real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1, 1, 1)
        alpha = alpha.to(self.device)
        alpha = alpha.expand_as(real_data)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        interpolates = interpolates + 1e-8
        disc_interpolates = self.Discriminator(interpolates)
        # gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device) \
                                      if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1).norm(2, dim=1) - 1) ** 2)
        #gradient_penalty = torch.mean(
         #   (1. - torch.sqrt(1e-8 + torch.sum(gradients.reshape(gradients.size(0), -1) ** 2, dim=1))) ** 2)
        return gradient_penalty

    # compute w distance
    def Wasserstein_Loss(self, labels, predictions):
        return torch.mean(labels * predictions)


    # load enhacner classifier
    def load_Enhancer_Classifier(self):

        args = parameter_parser2()

        # load the enhancer classifier class
        model = EnhancerClassifier(args).to(self.device)

        # load its state dictionary
        model.load_state_dict(torch.load("model"))
        model.eval()
        
        
        # return the model
        return model

    

    # predict whether that generated sequence is enhancer or not
    def predict_Enhancer_Sequence(self, epoch):
        self.predictions = []
        for i in tqdm(range(len(self.gen_seqs_list))):
            temp = self.gen_seqs_list[i].replace("P","")
            p = self.tokenize_seq(temp)
            q = torch.tensor(p)
            q = q.long()
            q = q.to(self.device)
            q.resize_((1,len(temp)))
            
            if not q.nelement() == 0 :
                prob = self.t(q).item()
                self.predictions.append(prob)
                if (prob > 0.7):
                    #print("Successfully generated enhancer with probability", prob)
                    #print(temp,"\n")
                
                    self.replace_enhancers(temp) ## check why this is commented out
                
                    # write the sequences to the files
                    with open('generated_sequences.txt', 'a') as f:
                        f.writelines(["P = ", str(prob), "\n", "E = ", str(epoch) , "\n", temp, "\n" ])
                                

    
    # convert sequence to tokens        
    def tokenize_seq(self, seq):
        sequence = list()
        dna =	{
                        "A": 1,
                        "T": 0,
                        "G": 3,
                        "C": 2 
                    }
        for i in range (len(seq)):
            sequence.append(dna[seq[i]])
        return sequence
        


    # function 2 to check validity of data (written by Jan Rottmayer)
    
    def check_data(self):
        
        # check in each row the data is like 1,0,0,0 or 0,1,0,0 or 0,0,1,0
        
        correct = []
        falseRows = []
        for i in range(len(self.data)):
            for j in range(300-1):
                control = self.data[i,j,:]
                flag = False
                for value in control:
                    if value == 1:
                        if flag:
                            falseRows.append((i,j))
                            correct.append(False)
                            break
                        flag = True
                if flag:
                    correct.append(True)
                else:
                    correct.append(False)
                    falseRows.append((i, j))
        return False if False in correct else True

    def plot_losses(self,g_loss, d_loss, gp_a, preds,epoch):

        plt.title("Discriminator's Loss")
        plt.plot(d_loss)
        plt.savefig('preds/'+ str(epoch)+'dloss.png')
        plt.clf()
        
        
        plt.title("Generator's Loss")
        plt.plot(g_loss)
        plt.savefig('preds/'+str(epoch)+'gloss.png')
        plt.clf()
        
        plt.title("Gradient Penalty")
        plt.plot(gp_a)
        plt.savefig('preds/'+str(epoch)+ 'gpa.png')
        plt.clf()
        
        plt.title("Classifier Predictions")
        plt.hist(preds)
        plt.savefig('preds/'+  str(epoch) + 'preds.png')
        plt.clf()
        
        
    # sample data from the generator
    def sample_generator(self):
        new_latent = torch.randn(self.batch_size,128).to(self.device)
        self.Generator.eval()
        a = self.Generator(new_latent)
        return a 
    
    # remove old indices from the real data
    def replace_enhancers(self, temp):
        
        # reset index value to 0 if it reaches to the maximum
        if (self.replace_index == len(self.data)):
            self.replace_index = 0
            
        # replace the index value in the original data
        self.data[self.replace_index]  = self.preprocessing.single_sequenceToOneHotEncoding(temp)
        self.replace_index += 1
        

        
        

    # decode the genrated sequences
    def hot_one(self, batch): 
        # Input is GPU Tensor
        batch = batch.cpu().detach().numpy()
        sequence_list = []
        #A = np.array([1,0,0,0,0])
        for seq in batch:
            sequence = ''
            for one_hot in seq:
                n = np.argmax(one_hot)
                if n==0 and one_hot[1] < one_hot[0] and one_hot[2] < one_hot[0] and one_hot[3] < one_hot[0] and one_hot[4] < one_hot[0]:
                    sequence += 'T'
                elif n==1:
                    sequence += 'A'
                elif n==2 :
                    sequence += 'C'
                elif n==3 :
                    sequence += 'G'
                elif n==4 :
                    sequence += 'P'
            sequence_list.append(sequence)
        self.gen_seqs_list = sequence_list
        

                
    # train the wgan-gp
    def train_WEnhancerGAN(self):

        #loader = DataLoader(self.data, batch_size=self.batch_size, drop_last=True)
        #print("data converted to loader")
        
        # create log lists
        self.g_loss_a = []
        self.d_loss_a = []
        self.w_dist_a = []
        self.gp_a=[]
        self.g_loss_epochs = []
        self.d_loss_epochs = []
        self.w_dist_epochs = []
        self.gp_epochs = []
        #torch.cuda.empty_cache()

        num_batches = int(len(self.data) / self.batch_size)


        # define the epochs
        # epochs = 50
        latent_dimensions = 128

        for epoch in tqdm(range(self.epochs)):
            
            print("Training epoch = ", epoch+1)
        
            start_index = 0 
            end_index = 0
            
            for i in range(1,num_batches+1):
                if (i <= num_batches):

                    end_index = self.batch_size * i
                    data = self.data[start_index : end_index]
                    start_index = start_index + self.batch_size
                    
                if (i == num_batches):
                    temp = (len(self.data[end_index:]))
                    temp = self.batch_size - temp
                    data =self.data [end_index - temp:]
                        
                # assign the labels
                real_labels = (torch.ones(self.batch_size)).to(self.device)
                fake_labels = - torch.ones(self.batch_size).to(self.device)


                ###### ** train the discriminator ** #######

                # avg discriminator loss
                d_loss_avg = 0

                # put real seqs to gpu
                real_seqs = torch.from_numpy(data).type(torch.FloatTensor).to(self.device)
                
                for _ in range (self.discriminator_steps):

                    # set the current gradient to zero
                    self.d_optim.zero_grad()

                    # generate sequences from the latent space
                    latent_vectors = torch.randn(self.batch_size, latent_dimensions).to(self.device)

                    fake_seqs = self.Generator(latent_vectors)

                    # score the sequences D(x)
                    real_score = self.Discriminator(real_seqs)

                    # probs on the fake data D(G(z))
                    fake_score = self.Discriminator(fake_seqs.detach())

                    # calculate the gradient penalty
                    gradient_penalty = self.Gradient_Norm(real_seqs, fake_seqs).mean()

                    # discirminator loss
                    d_loss =  -self.Wasserstein_Loss(real_labels, real_score) - self.Wasserstein_Loss(fake_labels, fake_score) + gradient_penalty * self.gpweight

                    # calc grads
                    d_loss.backward()

                    # apply the grads to the weights
                    self.d_optim.step()

                    # append the loss
                    d_loss_avg += d_loss

                    self.d_loss_a.append(d_loss.item())

                    self.gp_a.append(gradient_penalty.item())
                    
                    # weight clipping can be performed by the code below (not required here)
                    # for p in self.Discriminator.parameters():
                    #p.data.clamp_(-1, 1)

                ######  *** train the generator *** #####

                #set gradients to zero
                self.g_optim.zero_grad()

                # generate images from the latent space
                latent_vectors = torch.randn(self.batch_size, latent_dimensions).to(self.device)
                fake_seqs = self.Generator(latent_vectors)

                # scores on the fake seqs (D(G(z)))
                fake_s = self.Discriminator(fake_seqs)

                # coompute the w distance
                g_loss = self.Wasserstein_Loss(fake_labels, fake_s)

                # compute gradients
                g_loss.backward()

                # apply the gradients
                self.g_optim.step()

                # log the g loss
                self.g_loss_a.append(g_loss.item())

            print("Generator's Loss = ", self.g_loss_a[-1]/num_batches, "Discriminator's Loss:", self.d_loss_a[-1]/num_batches)
            self.g_loss_epochs.append(self.g_loss_a[-1]/num_batches)
            self.d_loss_epochs.append( self.d_loss_a[-1]/num_batches)
            #self.gp_epochs.append(gradient_penalty.item())
            self.hot_one(self.sample_generator())
            
            if (epoch%5==0):
                self.predict_Enhancer_Sequence(epoch)
                print("D",len(self.d_loss_epochs),"G", len(self.g_loss_epochs))
                self.plot_losses(self.g_loss_epochs, self.d_loss_epochs, self.gp_epochs,self.predictions, epoch)
                self.plot_losses(self.g_loss_a, self.d_loss_a, self.gp_a, self.predictions, epoch)
        with open("dloss.txt","wb") as fp:
            pickle.dump(self.d_loss_epochs,fp)
        with open("gloss.txt","wb") as fp:
            pickle.dump(self.g_loss_epochs,fp)
            
        # with open("dloss.txt", "rb") as fp:
        #     b = pickle.load(fp)
        
        # print(b)



args = parameter_parser()
wgan = WEnhancerGAN(args)

print(wgan.check_data())
wgan.train_WEnhancerGAN()
a = wgan.sample_generator()
wgan.hot_one(a)
torch.save(wgan.Generator.state_dict(), "Generator_model")
torch.save(wgan.Discriminator.state_dict(), "Discriminator_model")



'''
latent_vector = torch.randn(size=(128,)).to("cuda:3")
print(wgan.Generator(latent_vector))
'''







