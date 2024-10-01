#libraries required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import random
import time

#Code for PacketLSTM_GRU_D
class PacketLSTM_GRU_D(nn.Module):
    def __init__(self,n_class:int,n_features:int,device:None,lr:float=1e-4,hidden_size:int=64,\
        bias:bool=True,batch_size:int=1,aggregate_by='Mean',\
                feature_space:str = "Global",relu_in_prediction:bool=True,\
                decay:str='Decay',normalization:str='None',boundary:float=-0.01):
        super(PacketLSTM_GRU_D,self).__init__()
        """
        hidden_size- Output vector size of LSTM block
        n_features- maximum no of features
        n_class  - no of target classes
        bias - use bias or not [True,False]
        aggregate_by - aggregation function Choices - ['Mean','Max','Min','Sum','Unit_Vec']
        batch_size=1
        relu_in_prediction - Use ReLU in prediciton Stream or not [True,False]
        feature_space - Feature Space considered for prediction choices - ['Global','Current']
        normalization - Normalization Method Choices - ['None','Min_Max','Z_score','Mean_Norm','Unit_Vec','Dec_Scal']
        boundary - boundary for the weight constraint used in time_decay
        offline_imbalance- use offline Imbalance
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
        self.decay = decay
        #SHORT TERM MEMORY
        self.c_t = torch.zeros(self.hidden_size,dtype=torch.float32).to(self.device)
        #LONG TERM MEMORY is not stored globally as it is passed from instance to instance
        #Prediciton stream
        factory_args = {'dtype':torch.float32, 'device':self.device}
        # He's initialization for nn-parameters
        lstm_he_factor = math.sqrt(2/(self.hidden_size + 1))
        self.lstm_weights = nn.Parameter(lstm_he_factor * torch.randn(self.n_features, 4*self.hidden_size, self.hidden_size+1, **factory_args))
        self.lstm_bias = nn.Parameter(torch.zeros(self.n_features, 4*self.hidden_size, **factory_args))
        
        decay_he_factor = math.sqrt(2/(1))
        self.decay_weights = nn.Parameter(decay_he_factor * torch.randn(self.n_features, **factory_args))
        self.decay_bias = nn.Parameter(torch.zeros(self.n_features, **factory_args))
        
        in_for_out_dim =self.hidden_size + self.hidden_size
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
    def decay_func(self, feat_indices, delta, cur_device,Ht):
        feat_indices=feat_indices.to(cur_device)
        if self.decay=='Decay':
            tensor_zero = torch.tensor(0.).to(cur_device)
            decay_module_val = self.decay_weights[feat_indices]*(delta[feat_indices]).unsqueeze(dim=0)
            if self.bias:
                decay_module_val += self.decay_bias[feat_indices]
            decay = torch.exp(-torch.max(tensor_zero, decay_module_val))
            h_t_decayed = decay.unsqueeze(-1)*Ht[feat_indices,:]
        elif self.decay=='None':
            h_t_decayed = Ht[feat_indices,:].unsqueeze(0)
        else:
            exit()
        return h_t_decayed
    def forward(self,tim,X,X_hap,mask,Ht,Ct):
        self.time = time.time()
        agg_c_t = []
        agg_h_t = []
        H_missing = []
        H_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        C_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        self.feat_indices_curr = torch.arange(self.n_features).to(self.device)[mask==1]
        self.feat_indices_absent = torch.arange(self.n_features).to(self.device)[mask==0]
        self.feat_indices_new = torch.arange(self.n_features).to(self.device)[mask&(~self.feat_observed)]
        self.feat_indices_old = torch.arange(self.n_features).to(self.device)[mask&self.feat_observed]
        
        self.feat_observed = self.feat_observed | mask
        self.feat_indices_observed = torch.arange(self.n_features).to(self.device)[self.feat_observed==1]
        X_hap_t = self.normalize(X,mask,tim)
        delta = (tim - self.last_occured).to(device=self.device)
        Ht_decayed = self.decay_func(self.feat_indices_curr,delta,self.device,Ht)
        self.last_occured = tim*mask + (1^mask)*self.last_occured
        cur_input = torch.cat([X_hap_t.unsqueeze(-1), Ht_decayed.squeeze(0)], dim=-1)
        cur_output = torch.matmul(self.lstm_weights[self.feat_indices_curr], cur_input.unsqueeze(-1).float()).squeeze(-1)
        if self.bias:
                cur_output = cur_output + self.lstm_bias[self.feat_indices_curr]
        gate_input, gate_forget, gate_output, gate_pre_c = cur_output.chunk(4, dim=-1)
        gate_input = torch.sigmoid(gate_input)
        gate_forget = torch.sigmoid(gate_forget)
        gate_output = torch.sigmoid(gate_output)
        gate_pre_c = torch.tanh(gate_pre_c)
        agg_c_t = gate_forget * self.c_t + gate_input * gate_pre_c
        H_curr[self.feat_indices_curr,:] = gate_output * torch.tanh(agg_c_t)
        C_curr = self.agg_func(agg_c_t, self.aggregate_by)
        H_curr = torch.where(mask.any().unsqueeze(-1), H_curr, Ht)
        if self.feature_space=='Current':
            input_for_out = torch.reshape(torch.cat([C_curr,self.agg_func(H_curr[self.feat_indices_curr,:], self.aggregate_by)]),[-1])
        elif self.feature_space=='Universal':
            input_for_out = torch.reshape(torch.cat([C_curr,self.agg_func(H_curr[self.feat_indices_observed,:], self.aggregate_by)]),[-1])
        if self.relu_in_prediction:
            pred = torch.softmax(self.mlp(input_for_out),dim=0)
        else:
            pred = torch.softmax(self.output_layers(input_for_out),dim=0)
        with torch.no_grad():
          self.prediction.append(torch.argmax(pred).detach().cpu().item())
          self.pred_logits.append(pred[1].detach().cpu().numpy())
        return pred,H_curr,C_curr
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
        C_t_prev = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        
        counter = torch.zeros(self.n_class,dtype=torch.int,device=self.device)
        criterion = criterion().to(self.device)
        weights = torch.ones(self.n_class)
        for t in tqdm(range(X.shape[0])):
            optimizer.zero_grad()
            Y_predicted, H_t_curr, C_t_curr = self.forward(t+1,X[t].float(),X_hap[t].float(),mask[t],H_t_prev,C_t_prev)
            counter[Y[t].view(self.batch_size)]+=1
            loss = criterion(Y_predicted.view(self.batch_size, self.n_class), Y[t].view(self.batch_size).long())
            loss.backward(retain_graph=True)
            optimizer.step()
            H_t_curr = H_t_curr.detach()
            C_t_curr = C_t_curr.detach()
            #self.train_losses.append(loss.detach().item())
            H_t_prev = H_t_curr
            C_t_prev = C_t_curr
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
                self.m[self.feat_indices_curr] = X[self.feat_indices_curr].float()
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