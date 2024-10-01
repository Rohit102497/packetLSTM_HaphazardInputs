#libraries required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import random
import time

#Code for PacketGRU
class PacketGRU(nn.Module):
    def __init__(self,n_class:int,n_features:int,device:None,lr:float=1e-4,hidden_size:int=64,\
        bias:bool=True,batch_size:int=1,aggregate_by='Mean',\
                feature_space:str = "Current",relu_in_prediction:bool=True, memory_type='Both',\
                decay:str='None', normalization:str='None',boundary:float=-0.01):
        super(PacketGRU,self).__init__()
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
        boundary - boundary for the weight constraint used in time_decay
        memory_type - Which memory to use for prediction stream ['LTM','STM','Both']
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
        self.boundary = boundary 
        self.lr = lr
        self.memory_type = memory_type
        
        factory_args = {'dtype':torch.float32, 'device':self.device}
        #GRU weights
        gru_xt_he_factor = math.sqrt(2/(1))
        self.gru_xT_weights = nn.Parameter(gru_xt_he_factor * torch.randn(self.n_features,3*self.hidden_size,1,**factory_args))
        self.gru_xT_bias = nn.Parameter(torch.zeros(self.n_features,3*self.hidden_size,**factory_args))
        
        
        gru_U_he_factor = math.sqrt(2/(self.hidden_size))        
        self.gru_U_weights = nn.Parameter(gru_U_he_factor * torch.randn(self.n_features,3*self.hidden_size,self.hidden_size,**factory_args))
        
        in_for_out_dim =self.hidden_size
        
        self.output_layers = nn.Linear(in_for_out_dim, self.n_class).to(self.device)
        self.mlp = nn.Sequential(
            nn.Linear(in_for_out_dim, in_for_out_dim),
            nn.ReLU(),
            nn.Linear(in_for_out_dim, self.n_class),
        )
        
        #Normalization
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
    
    def forward(self,tim,X,X_hap,mask,Ht):
        #tim is the instance no.
        #list of STM,LTM of each feature for aggregation
        H_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        C_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
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
        
        #GRU BLOCK
        
        x_T_out = torch.matmul(self.gru_xT_weights[self.feat_indices_curr],X_hap_t.unsqueeze(-1).unsqueeze(-1).float()).squeeze(-1)
        U_T_out = torch.matmul(self.gru_U_weights[self.feat_indices_curr],Ht[self.feat_indices_curr,:].unsqueeze(-1)).squeeze(-1)
        if self.bias:
            x_T_out = x_T_out + self.gru_xT_bias[self.feat_indices_curr]
        x_z, x_r , x_h = x_T_out.chunk(3,dim=-1)
        u_z, u_r, u_h = U_T_out.chunk(3,dim=-1)
        z_gate = torch.sigmoid(x_z + u_z)
        r_gate = torch.sigmoid(x_r + u_r)
        h_tilde_gate  = torch.tanh(x_h + r_gate * u_h)
        h_gate = z_gate*Ht[self.feat_indices_curr]+(1-z_gate)*h_tilde_gate
        
        H_curr[self.feat_indices_curr,:] = h_gate
        H_curr = torch.where(mask.any().unsqueeze(-1), H_curr, Ht)
        if self.feature_space=='Current':
            input_for_out = self.agg_func(H_curr[self.feat_indices_curr,:], self.aggregate_by)
        elif self.feature_space=='Universal':
            input_for_out = self.agg_func(H_curr[self.feat_indices_observed,:], self.aggregate_by)
            
        if self.relu_in_prediction:
            pred = torch.softmax(self.mlp(input_for_out),dim=0)
        else:
            pred = torch.softmax(self.output_layers(input_for_out),dim=0)
        with torch.no_grad():
          self.prediction.append(torch.argmax(pred).detach().cpu().item())
          self.pred_logits.append(pred[1].detach().cpu().numpy())
        return pred,H_curr
    
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
        if fn == "Mean":
            return torch.mean(tensors, dim=0)
        elif fn == 'Max':
            return torch.max(tensors, dim=0).values
        elif fn == 'Min':
            return torch.min(tensors,dim=0).values
        elif fn == 'Sum':
            return torch.sum(tensors,dim=0)