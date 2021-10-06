import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score



from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Import the neccessary files
from src_classifier.utils import Preprocessing
from src_classifier.model import EnhancerClassifier
from src_classifier.parser import parameter_parser

# class to handle batches
class DatasetMaper(Dataset):
    
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
		

class Execute:
    
    def __init__(self, args, train= False):
        if train:
            self.__init_data__(args)
        self.args = args
        self.batch_size = args.batch_size
        self.device = "cuda:3"
        self.model = EnhancerClassifier(args).to(self.device)
        self.x_train = None
        self.x_test = None
        
    def __init_data__(self, args):
        
        # function used from the utils files (see utils for details)
        self.preprocessing = Preprocessing(args)
        
        # read fasta of positive sequences
        self.preprocessing.load_data()
        
        # read fasta of negative sequences
        self.preprocessing.add_background_sequences()
        
        # subsample from the negative sequences
        self.preprocessing.optimize_negative_sequences()
        
        # preparing train and test data
        self.preprocessing.prepare_train_test()
        self.preprocessing.prepare_tokens()
        
        raw_x_train = self.preprocessing.x_train
        raw_x_test = self.preprocessing.x_test
        
        
        self.y_train = self.preprocessing.y_train
        self.y_test = self.preprocessing.y_test
        
        self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
        self.x_test = self.preprocessing.sequence_to_token(raw_x_test)
        
        
        
    def train(self):
        
        training_set = DatasetMaper(self.x_train, self.y_train)
        test_set = DatasetMaper(self.x_test, self.y_test)
        
        # initializing the dataloader 
        self.loader_training = DataLoader(training_set, batch_size=self.batch_size )
        self.loader_test = DataLoader(test_set, batch_size=self.batch_size )
        
        # defining the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate)#.to(device)
        
        # iterating from the epochs
        for epoch in tqdm(range(args.epochs)):

            # Clear gpu after every epoch
            torch.cuda.empty_cache()
            
            # the predictions
            predictions = []
            self.model.train()
            
            # epoch loop
            for x_batch, y_batch in self.loader_training:
                
                # handling x and y batches + sending to cuda
                x = x_batch.type(torch.LongTensor).to(self.device)
                y = y_batch.type(torch.FloatTensor).to(self.device)
                
                # make the predictions
                y_pred = self.model(x)
                
                # compute the loss
                loss = F.binary_cross_entropy(y_pred, y.unsqueeze(1))
                
                # reseting the grads
                optimizer.zero_grad()
                
                # compute gradients
                loss.backward()
                
                # update the parameters
                optimizer.step()
                
                # appending prediction list with singular predictions
                for pred in y_pred.squeeze():
                    predictions.append(pred)
                    
            # call the evaluation function        
            test_predictions = self.evaluationCUDA() 
                        
            # compute train test accuracy
            train_accuary = self.calculate_accuray(self.y_train, predictions)
            test_accuracy = self.calculate_accuray(self.y_test, test_predictions)
            
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))

    
    def evaluationCUDA(self):
        predictions = []

        # model.eval() will notify all our layers that we are in eval mode
        self.model.eval()
        
        # no computation of grads
        with torch.no_grad():
            
            # iterate over dataloader object
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.LongTensor).to(self.device)
                y = y_batch.type(torch.FloatTensor).to(self.device)
                
                
                # feed the model
                y_pred = self.model(x)
                
                # save predictions
                for pred in y_pred.squeeze():
                    predictions.append(pred)
        return predictions
    
    # compute accuracy
    @staticmethod # bouind with class not with object
    def calculate_accuray(grand_truth, predictions):
        true_positives = 0
        true_negatives = 0
        for true, pred in zip(grand_truth, predictions):
            if (pred > 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass
        print("tp:", true_positives, "tn:", true_negatives)
        print("g t = ", len(grand_truth))
        return (true_positives+true_negatives) / len(grand_truth)
    
    
    '''

      # compute accuracy
    @staticmethod # bouind with class not with object
    def calculate_accurayCUDA(grand_truth, predictions):
        true_positives = 0
        true_negatives = 0
        for true, pred in zip(torch.Tensor(grand_truth).to("cuda:3"), predictions):
            if (pred > 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass
        return (true_positives+true_negatives) / len(grand_truth)
    # evaluation function o ri n
    def evaluation(self):
        predictions = []
        
        self.model.eval() #Signaling no Training!
        
        # no computation of grads
        with torch.no_ [0.7682125 ]
grad():
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.LongTensor).to(self.device)
                y = y_batch.type(torch.FloatTensor).to(self.device)
                y_pred = self.model(x)
                predictions += list(y_pred.detach().cpu().clone().numpy())
        return predictions
    '''

	
'''
def my_plot(epochs, loss):
    plt.plot(epochs, loss)
    
def train(num_epochs,optimizer,criterion,model):
    loss_vals=  []
    for epoch in range(num_epochs):
        epoch_loss= []
        for i, (images, labels) in enumerate(trainloader):
            # rest of the code
            loss.backward()
            epoch_loss.append(loss.item())
            # rest of the code
        # rest of the code
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))
        # rest of the code
    
    # plotting
    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals)
'''
	
args = parameter_parser()	
execute = Execute(args, train=False)
#execute.train()
#print("\ntrain data shape:",execute.x_train.shape)
#print("\ntest data shape:", execute.x_test.shape)







p = torch.tensor([2,4,2,4,2,2,4,3,2,4,2,4,3,4,3,1,3,3,4,3,3,3,2,2,2,4,3,3,2,4,3,3,1,3,3,4,1,1,3,3,3,1,3,1,2,2,4,2,4,4,
4,3,1,4,3,2,4,3,3,3,2,1,1,1,3,3,1,2,4,1,3,1,3,3,2,4,2,4,1,1,3,2,4,4,2,2,1,4,3,4,2,4,2,2,1,4,4,4,3,2,
4,3,3,3,4,3,1,3,3,3,4,1,3,3,4,4,3,3,1,2,2,1,4,3,4,3,1,3,1,4,3,4,4,1,3,3,1,1,3,1,3,4,3,4,3,3,4,3,3,1,
3,3,2,2,3,2,1,2,2,2,4,3,3,3,2,4,4,2,2,2,2,2,4,2,2,1,3,3,3,2,2,2,4,2,2,2,4,3,3,1,2,1,3,4,2,2,4,3,3,2,
2,1,3,4,4,2,4,3,4,3,1,2,2,4,3,4,4,4,1,3,4,3,1,4,3,4,2,2,2,4,1,4,2,1,2,2,1,4,2,1,2,2,2,1,3,3,3,1,3,1,
1,3,2,1,3,1,1,1,2,3,1,1,1,2,4,3,3,3,4,3,4,1,1,2,2,2,3,1,1,1,2,3,3,1,1,2


]).to("cuda:3")

q = p.unsqueeze(0)
##for i in range(10):
    #print(execute.model(q).shape)
  #  print(q.shape)



#torch.save(execute.model.state_dict(),"model")

#t = EnhancerClassifier(args).to("cuda:3")
#t.load_state_dict(torch.load("model"))
#for i in range(10):
    #print(t(q))



