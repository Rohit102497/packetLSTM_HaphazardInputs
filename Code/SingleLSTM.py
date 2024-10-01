#libraries required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import torch.nn.functional as F

#Code for PacketTimeLSTM_3
class SingleLSTM(nn.Module):
    def __init__(self,n_class:int,n_features:int,device:None,exp_type:str=None, lr:float=1e-4, hidden_size:int=64,\
                batch_size:int=1,\
                relu_in_prediction:bool=True,\
                normalization:str='None',
                num_layers:int=1,
                dropout:float=0.0):
        super(SingleLSTM,self).__init__()
        """
        hidden_size- Output vector size of LSTM block
        n_features- maximum no of features
        n_class  - no of target classes
        bias - use bias or not [True,False]
        aggregate_by - aggregation function Choices - ['Mean','Max','Min','Sum','Unit_Vec']
        batch_size=1
        relu_in_prediction - Use ReLU in prediciton Stream or not [True,False]
        feature_space - Feature Space considered for prediction choices - ['Current','Universal']
        normalization - Normalization Method Choices - ['None','Min_Max','Z_score','Mean_Norm','Unit_Vec','Dec_Scal']
        memory_type - Which memory to use for prediction stream ['LTM','STM','Both']
        lr - learning rate
        """
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.device = device
        self.n_class = n_class
        self.relu_in_prediction=relu_in_prediction
        self.batch_size=int(batch_size)

        self.num_layers = num_layers
        self.dropout = dropout
        self.exp_type = exp_type

        if self.exp_type == 'Imputation':
            self.lstm = nn.LSTM(self.n_features, self.hidden_size, num_layers, dropout = dropout).to(self.device)
        self.output_layers = nn.Linear(self.hidden_size, self.n_class).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_class),
        )
        
        self.normalization = normalization
        self.lr = lr
        
        self.count = 0
        #Performance Evaluation
        self.prediction = []
        self.train_losses=[]
        self.pred_logits=[]

    
    def forward(self, X, mask):
        if self.exp_type == 'Imputation':
            input_for_out, _ = self.lstm(torch.unsqueeze(X, 0))
                
        if self.relu_in_prediction:
            pred = torch.softmax(self.mlp(input_for_out),dim=1)
        else:
            pred = torch.softmax(self.output_layers(input_for_out),dim=1)

        with torch.no_grad():
          self.prediction.append(torch.argmax(pred).detach().cpu().item())
          self.pred_logits.append(pred[0][1].detach().cpu().numpy())
        return pred
    
    def fit(self, X, Y, mask):
        self.prediction = []
        #self.train_losses=[]
        self.pred_logits=[]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss
        criterion = criterion().to(self.device)
        
        # Input Data
        X=torch.tensor(X).to(self.device)
        Y=torch.tensor(Y).to(self.device,dtype=torch.int)
        mask=torch.tensor(mask,dtype=torch.bool).to(self.device)
        counter = torch.zeros(self.n_class,dtype=torch.int,device=self.device)
        
        print(X.shape[0])
        for t in tqdm(range(X.shape[0])):
            optimizer.zero_grad()
            self.count += 1
            Y_predicted = self.forward(X[t].float(), mask[t])
            counter[Y[t].view(self.batch_size)]+=1
            loss = criterion(Y_predicted.view(self.batch_size, self.n_class), Y[t].view(self.batch_size).long())
            loss.backward(retain_graph=True)
            optimizer.step()