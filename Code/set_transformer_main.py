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
from set_transformer.modules import SAB, PMA, ISAB

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Code for Transformer
class set_transformer(nn.Module):
    def __init__(self, n_class:int, n_features:int,\
        device='cuda', batch_size:int=1,
        hidden_size:int=512,
        num_inds=32, n_heads=4,
        normalization:str='Z_score',
        lr:float=0.0006):
        super(set_transformer,self).__init__()
        """
        hidden_size- Output vector size of LSTM block
        """
        self.n_features = n_features
        self.n_class = n_class
        self.device = device

        self.batch_size = batch_size
        self.normalization = normalization
        self.lr = lr

        self.normalization = 'Z_score'
        self.m = torch.zeros(self.n_features).to(device=self.device)
        self.v = torch.zeros(self.n_features).to(device=self.device)
        
        dim_input=1 
        num_outputs=1
        ln=False

        self.enc = nn.Sequential(
                ISAB(dim_input, hidden_size, n_heads, num_inds, ln=ln),
                ISAB(hidden_size, hidden_size, n_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(hidden_size, n_heads, num_outputs, ln=ln),
                SAB(hidden_size, hidden_size, n_heads, ln=ln),
                # SAB(hidden_size, hidden_size, num_heads, ln=ln),
                nn.Linear(hidden_size, n_class))
        
        # Performance Evaluation
        self.prediction = []
        self.train_losses=[]
        self.pred_logits=[]
        self.count = 0
     
    def forward(self,tim,X_hap,mask):
        self.feat_indices_curr = torch.arange(self.n_features).to(self.device)[mask==1]
        self.feat_indices_absent = torch.arange(self.n_features).to(self.device)[mask==0]
        self.feat_indices_new = torch.arange(self.n_features).to(self.device)[mask&(~self.feat_observed)]
        self.feat_indices_old = torch.arange(self.n_features).to(self.device)[mask&self.feat_observed]
        self.feat_count[self.feat_indices_curr]+=1
        self.feat_observed = self.feat_observed | mask
        X_hap_normalized = self.normalize(tim, X_hap) #.reshape(-1,1)

        # Creating inputs of shape [1, number of observed feautres, 1]
        inp = torch.unsqueeze(torch.unsqueeze(X_hap_normalized[mask], 0), 2)

        enc_out = self.enc(inp)
        dec_out = self.dec(enc_out)
        
        pred = torch.softmax(torch.squeeze(dec_out, 0), dim = 1)
        pred = pred.reshape(-1)
        with torch.no_grad():
          self.prediction.append(torch.argmax(pred).detach().cpu().item())
          self.pred_logits.append(pred[1].detach().cpu().numpy())
        
        self.time = time.time()
        return pred
    
    def fit(self, X_hap, Y, mask):
        
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