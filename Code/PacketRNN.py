#libraries required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import random
import time

# Code for PacketRNN
class PacketRNN(nn.Module):
    def __init__(self,n_class:int,n_features:int,device:None,lr:float=1e-4,hidden_size:int=64,\
        bias:bool=True,batch_size:int=1,aggregate_by='Mean',\
                feature_space:str = "Current",relu_in_prediction:bool=True,\
                decay:str='None',normalization:str='None',boundary:float=-0.01):
        super(PacketRNN,self).__init__()
        """
        hidden_size- Output vector size of LSTM block
        n_features- maximum no of features
        n_class  - no of target classes
        bias - use bias or not [True,False]
        decay - 'Decay','None'
        aggregate_by - aggregation function Choices - ['Mean','Max','Min','Sum','Unit_Vec']
        batch_size=1
        relu_in_prediction - Use ReLU in prediciton Stream or not [True,False]
        feature_space - Feature Space considered for prediction choices - ['Current','Universal']
        normalization - Normalization Method Choices - ['None','Min_Max','Z_score','Mean_Norm','Unit_Vec','Dec_Scal']
        boundary - boundary for the weight constraint used in time_decay
        """
        self.hidden_size = hidden_size #HiddenSize of LSTMCELL
        self.n_features = n_features
        self.n_class = n_class
        self.bias=bias
        self.aggregate_by = aggregate_by
        self.device = device
        self.batch_size=int(batch_size)
        self.relu_in_prediction=relu_in_prediction
        self.feature_space=feature_space
        self.normalization = normalization
        self.lr = lr
        self.decay = decay
        #SHORT TERM MEMORY
        self.H_t = torch.zeros(self.hidden_size,dtype=torch.float32).to(self.device)
        
        factory_args = {'dtype':torch.float32, 'device':self.device}
        # RNN BLOCK weights
        rnn_he_factor = math.sqrt(2/(1))
        self.Wx = nn.Parameter(rnn_he_factor * torch.randn(self.n_features,self.hidden_size,1,**factory_args))
        self.Wc = nn.Parameter(rnn_he_factor * torch.randn(self.n_features,self.hidden_size,self.hidden_size,**factory_args))
        self.rnn_bias = nn.Parameter(torch.zeros(self.n_features,self.hidden_size,**factory_args))
        
        # Decay Block
        decay_he_factor = math.sqrt(2/(1))
        self.decay_weights = nn.Parameter(decay_he_factor * torch.randn(self.n_features, **factory_args))
        self.decay_bias = nn.Parameter(torch.zeros(self.n_features, **factory_args))
        
        #Prediction network
        in_for_out_dim =self.hidden_size
        self.output_layers = nn.Linear(in_for_out_dim, self.n_class).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(in_for_out_dim, in_for_out_dim),
            nn.ReLU(),
            nn.Linear(in_for_out_dim, self.n_class),
        )
        
        #Initaliation for Normalization
        if normalization=='Min_Max':
            self.min = torch.zeros(self.n_features)
            self.max = torch.zeros(self.n_features)
        if normalization=='Mean_Norm' or normalization=='Z_score':
            #https://math.stackexchange.com/a/116344
            self.m = torch.zeros(self.n_features)
            self.v = torch.zeros(self.n_features)
        #Performance Evaluation
        self.prediction = []
        self.train_losses=[]
        self.pred_logits=[]
    def decay_func(self, feat_indices, delta, cur_device,Ht):
        ## There won't be any decay for this code
        feat_indices=feat_indices.to(cur_device)
        if self.decay=='None':
            h_t_decayed = Ht[feat_indices,:].unsqueeze(0)
        else:
            print("Please set decay as None in config")
            exit()
        return h_t_decayed
    def forward(self,tim,X,X_hap,mask,Ht):
        #tim is the instance no.
        # Ht is the previous STM of RNN block
        #H_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        H_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        # Feature information
        # Feature indices. of current features
        self.feat_indices_curr = torch.arange(self.n_features).to(self.device)[mask==1]
        #indices of newly appeared features
        self.feat_indices_new = torch.arange(self.n_features).to(self.device)[mask&(~self.feat_observed)]
        #indices of current features which were already observed
        self.feat_indices_old = torch.arange(self.n_features).to(self.device)[mask&self.feat_observed]
        #count of current features
        self.feat_count[self.feat_indices_curr]+=1
        # Track of features which were observed till now
        self.feat_observed = self.feat_observed | mask
        #and their indices
        self.feat_indices_observed = torch.arange(self.n_features).to(self.device)[self.feat_observed==1]
        #last_occured - latest time instance they appeared -> if mask=1 tim else self.last_occured
        self.last_occured = tim*mask + (1^mask)*self.last_occured
        
        # Normalization of Input Data
        X_hap_t = self.normalize(X,mask,tim)
        
        delta = (tim - self.last_occured).to(device=self.device)
        Ht_decayed = self.decay_func(self.feat_indices_curr,delta,self.device,Ht)
        # RNN block
        x_T_out = torch.matmul(self.Wx[self.feat_indices_curr],X_hap_t.unsqueeze(-1).unsqueeze(-1).float()).squeeze(-1)
        h_T_out = torch.matmul(self.Wc[self.feat_indices_curr],Ht_decayed.unsqueeze(-1).float()).squeeze(-1)
        if self.bias:
           x_T_out = x_T_out + self.rnn_bias[self.feat_indices_curr] 
        Ht[self.feat_indices_curr] = torch.tanh(x_T_out + h_T_out)    
        
        # Aggregation of STM   
        if self.feature_space=='Current':
            input_for_out = self.agg_func(Ht[self.feat_indices_curr,:],self.aggregate_by).squeeze(-1)
        elif self.feature_space=='Universal':
            input_for_out = torch.agg_func(Ht[self.feat_indices_observed,:],self.aggregate_by).squeeze(-1)
            
        # Prediction stream
        if self.relu_in_prediction:
            pred = torch.softmax(self.mlp(input_for_out),dim=0)
        else:
            pred = torch.softmax(self.output_layers(input_for_out),dim=0)
        with torch.no_grad():
          self.prediction.append(torch.argmax(pred).detach().cpu().item())
          self.pred_logits.append(pred[1].detach().cpu().numpy())
        return pred, H_curr
    def fit(self,X,X_hap,Y,mask):
        self.prediction = []
        #self.train_losses=[]
        self.pred_logits=[]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss
        
        # Input Data
        X=torch.tensor(X).to(self.device)
        X_hap=torch.tensor(X_hap).to(self.device)
        Y=torch.tensor(Y).to(self.device,dtype=torch.int)
        mask=torch.tensor(mask,dtype=torch.bool).to(self.device)
        self.feat_observed = torch.zeros(self.n_features,dtype=torch.bool,device=self.device)
        self.last_occured = torch.zeros(self.n_features,dtype=torch.int,device=self.device)
        self.feat_count = torch.zeros(self.n_features,dtype=torch.int,device=self.device)
        # Initalize LTM,STM
        H_t_prev = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        
        counter = torch.zeros(self.n_class,dtype=torch.int,device=self.device)
        criterion = criterion().to(self.device)
        weights = torch.ones(self.n_class)
        for t in tqdm(range(X.shape[0])):
            optimizer.zero_grad()
            Y_predicted, H_t_curr = self.forward(t+1,X[t].float(),X_hap[t].float(),mask[t],H_t_prev)
            counter[Y[t].view(self.batch_size)]+=1
            loss = criterion(Y_predicted.view(self.batch_size, self.n_class), Y[t].view(self.batch_size).long())
            loss.backward(retain_graph=True)
            optimizer.step()
            H_t_curr = H_t_curr.detach()
            #self.train_losses.append(loss.detach().item())
            H_t_prev = H_t_curr
            #print(loss.detach().item())
    def normalize(self,X,mask,tim):
        # Normalization
        X_hap_t = X
        if self.normalization=='None':
            X_hap_t = X[self.feat_indices_curr]
        elif self.normalization=='Min_Max':
            if tim==1:
                self.min[self.feat_indices_curr] = X[self.feat_indices_curr].float()
                self.max[self.feat_indices_curr] = X[self.feat_indices_curr].float()
            else:
                self.min[self.feat_indices_new] = X[self.feat_indices_new].float()
                self.max[self.feat_indices_new] = X[self.feat_indices_new].float()
                self.min[self.feat_indices_old] = torch.minimum(X[self.feat_indices_old].float(),self.min[self.feat_indices_old].float())
                self.max[self.feat_indices_old] = torch.maximum(X[self.feat_indices_old].float(),self.max[self.feat_indices_old].float())
                if len(self.feat_indices_old)>0:
                    if torch.min((self.max[self.feat_indices_old]-self.min[self.feat_indices_old]))>0.0:
                        X[self.feat_indices_old] = (X[self.feat_indices_old]-self.min[self.feat_indices_old])/(self.max[self.feat_indices_old]-self.min[self.feat_indices_old])
        elif self.normalization=='Z_score':
            if tim==1:
                self.m[self.feat_indices_curr] = X[self.feat_indices_curr]
            else:
                self.m[self.feat_indices_new] = X[self.feat_indices_new].float()
                count = self.feat_count[self.feat_indices_old]
                m_t = self.m[self.feat_indices_old]+(X[self.feat_indices_old]-self.m[self.feat_indices_old])/count
                self.v[self.feat_indices_old] = self.v[self.feat_indices_old]+(X[self.feat_indices_old]-self.m[self.feat_indices_old])*(X[self.feat_indices_old]-m_t)
                self.m[self.feat_indices_old] = m_t
                if len(self.feat_indices_old)>0:
                    if torch.min(self.v[self.feat_indices_old])>0.0:
                        X[self.feat_indices_old] = (((X[self.feat_indices_old]-self.m[self.feat_indices_old])).float()/torch.sqrt(self.v[self.feat_indices_old]/(count-1)))
        elif self.normalization=='Dec_Scal':
            X[self.feat_indices_curr] = X[self.feat_indices_curr]/pow(10,3) # Power can be choosen
        elif self.normalization=='Unit_Vec':
            X[self.feat_indices_curr] = torch.nn.functional.normalize(X[self.feat_indices_curr],dim=0)
        elif self.normalization=='t_Digest':
            pass
        elif self.normalization=='Mean_Norm':
            if tim==1:
                self.m[self.feat_indices_curr] = X[self.feat_indices_curr]
            else:
                count = self.feat_count[self.feat_indices_old]
                self.m[self.feat_indices_new] = X[self.feat_indices_new].float()
                m_t = self.m[self.feat_indices_old]*((count-1)/count)+(X[self.feat_indices_old]/count)
                X[self.feat_indices_old] = ((X[self.feat_indices_old]-self.m[self.feat_indices_old])).float() 
        return X[self.feat_indices_curr]
    def agg_func(self,tensors, fn='Mean'):
        ## Aggregation Function
        if fn == "Mean":
            return torch.mean(tensors, dim=0)
        elif fn == 'Max':
            return torch.max(tensors, dim=0).values
        elif fn == 'Min':
            return torch.min(tensors,dim=0).values
        elif fn == 'Sum':
            return torch.sum(tensors,dim=0)