# https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py

#libraries required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import random
import time
import os
import torch.nn.functional as F

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Code for Transformer
class online_transformer(nn.Module):
    def __init__(self, n_class:int, n_features:int,\
        device='cuda', exp_type:str='PaddedInputs' ,batch_size:int=1, n_heads:int=8,
        hidden_size:int=512, n_layers:int=1,
        dropout:float=0.0, normalization:str='Z_score',
        relu_in_prediction:bool=True, lr:float=0.0006):
        super(online_transformer,self).__init__()
        """
        hidden_size- Output vector size of LSTM block
        """
        self.n_features = n_features
        self.n_class = n_class
        self.device = device

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.normalization = normalization
        self.lr = lr
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.normalization = 'Z_score'
        self.m = torch.zeros(self.n_features).to(device=self.device)
        self.v = torch.zeros(self.n_features).to(device=self.device)
        self.relu_in_prediction = relu_in_prediction
        self.exp_type = exp_type

        #Prediciton stream
        factory_args = {'dtype':torch.float32, 'device':self.device}
        if self.exp_type == 'PaddedInputs':
            self.element_embedding = nn.Linear(self.n_features, self.hidden_size, bias=True,device=self.device)
        else:
            self.element_embedding = nn.Linear(2 * self.n_features, self.hidden_size, bias=True,device=self.device)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.hidden_size, 
                                                                            self.n_heads,
                                                                            self.hidden_size,
                                                                            self.dropout,
                                                                            batch_first=True,
                                                                            **factory_args), self.n_layers).to(device=self.device)
        in_for_out_dim = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(in_for_out_dim, in_for_out_dim),
            nn.ReLU(),
            nn.Linear(in_for_out_dim, self.n_class),
        ).to(device=self.device)
        self.output_layers = nn.Linear(in_for_out_dim, self.n_class).to(device = self.device)
        
        # Performance Evaluation
        self.prediction = []
        self.train_losses=[]
        self.pred_logits=[]
        self.count = 0
        if self.exp_type == 'InputPairs':
            self.mask_index = torch.arange(start = 1, end = self.n_features+1).to(device = self.device)
     
    def forward(self,tim,X_hap,mask):
        self.feat_indices_curr = torch.arange(self.n_features).to(self.device)[mask==1]
        self.feat_indices_absent = torch.arange(self.n_features).to(self.device)[mask==0]
        self.feat_indices_new = torch.arange(self.n_features).to(self.device)[mask&(~self.feat_observed)]
        self.feat_indices_old = torch.arange(self.n_features).to(self.device)[mask&self.feat_observed]
        self.feat_count[self.feat_indices_curr]+=1
        self.feat_observed = self.feat_observed | mask
        X_hap_normalized = self.normalize(tim, X_hap) #.reshape(-1,1)
        
        if self.exp_type == 'InputPairs':
            '''
            Here we passed the normalized haphazard inputs without imputation. Only the non nan values are considered.
            We also pass the mask as feature index. For this the feature index is numbered from 1 till number of features.
            The non nan haphazard inputs and the available feature indices are concatenated in an alternate fashion (scissor way).
            Finally, the input is padded with zeros to match the length of 2*(number of features).
            Inputs looks like: [value1, f1, value2, f2, ..., pad] with shape 2*(number of features)
            '''
            emb_inp = torch.cat([X_hap_normalized[mask].unsqueeze(1), self.mask_index[mask].unsqueeze(1)], dim=-1).view(-1, 1)
            emb_inp = F.pad(emb_inp, pad = (0, 0, 0, 2*self.n_features - emb_inp.shape[0])).squeeze(1)
            transformer_input = torch.unsqueeze(self.element_embedding(emb_inp), 0)
        elif self.exp_type == 'PaddedInputs':
            input_padded = torch.zeros(self.n_features).to(device = self.device)
            input_padded[:torch.sum(mask)] = X_hap_normalized[mask]
            # print(X_hap_normalized, mask, "input_padded: ", input_padded)
            transformer_input = torch.unsqueeze(self.element_embedding(input_padded), 0)
        transformer_output = self.transformer(transformer_input)
        
        if self.relu_in_prediction:
            pred = torch.softmax(self.mlp(transformer_output),dim=1)
        else:
            pred = torch.softmax(self.output_layers(transformer_output),dim=1)
            
        pred = pred.reshape(-1)
        with torch.no_grad():
          self.prediction.append(torch.argmax(pred).detach().cpu().item())
          self.pred_logits.append(pred[1].detach().cpu().numpy())
        
        self.time = time.time()
        return pred
    
    def fit(self,X,X_hap,Y,mask):
        
        self.prediction = []
        self.pred_logits=[]
        X_hap=torch.tensor(X_hap).to(self.device)
        Y=torch.tensor(Y).to(self.device,dtype=torch.int)
        mask=torch.tensor(mask,dtype=torch.bool).to(self.device)
        self.feat_observed = torch.zeros(self.n_features,dtype=torch.bool,device=self.device)
        self.last_occured = torch.zeros(self.n_features,dtype=torch.int,device=self.device)
        self.feat_count = torch.zeros(self.n_features,dtype=torch.int,device=self.device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss #().to(self.device)
        criterion_fn = criterion(reduction='none').to(self.device)
        
        for t in tqdm(range(X_hap.shape[0])):
            optimizer.zero_grad()
            Y_predicted = self.forward(t+1, X_hap[t].float(),mask[t])
            loss = criterion_fn(Y_predicted.view(self.batch_size, self.n_class), Y[t].view(self.batch_size).long())
            loss.backward()
            optimizer.step()  
        
    def normalize(self,tim,X):
        if self.normalization == 'Z_score':
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
        return X