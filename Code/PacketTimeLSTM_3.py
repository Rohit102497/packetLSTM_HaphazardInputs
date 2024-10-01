#libraries required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from tqdm import tqdm

#Code for PacketTimeLSTM_3
class PacketTimeLSTM_3(nn.Module):
    def __init__(self,n_class:int,n_features:int,device:None,lr:float=1e-4,hidden_size:int=64,\
                bias:bool=True,batch_size:int=1,aggregate_by='Mean',\
                feature_space:str = "Current",relu_in_prediction:bool=True,\
                decay:str="TimeLSTM_3", normalization:str='None',boundary:float=-0.01,memory_type:str='Both'):
        super(PacketTimeLSTM_3,self).__init__()
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
        lr - learning rate
        """
        #HiddenSize of LSTMCELL
        self.hidden_size = hidden_size 
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
        #SHORT TERM MEMORY
        self.c_t = torch.zeros(self.hidden_size,dtype=torch.float32).to(self.device)
        #LONG TERM MEMORY is not stored globally as it is passed from instance to instance
        #Prediciton stream
        factory_args = {'dtype':torch.float32, 'device':self.device}
        # LSTM BLOCK weights
        lstm_he_factor = math.sqrt(2/(self.hidden_size + 1))
        self.lstm_weights = nn.Parameter(lstm_he_factor * torch.randn(self.n_features, 3*self.hidden_size, self.hidden_size+1, **factory_args))
        self.lstm_bias = nn.Parameter(torch.zeros(self.n_features, 3*self.hidden_size, **factory_args))
        #Decay Weights
        lstm_xt_he_factor = math.sqrt(2/(1))
        self.lstm_xT_weights = nn.Parameter(lstm_xt_he_factor * torch.randn(self.n_features,2*self.hidden_size,1,**factory_args))
        self.lstm_xT_bias = nn.Parameter(torch.zeros(self.n_features,2*self.hidden_size,**factory_args))
        
        lstm_del_he_factor = math.sqrt(2/(1))        
        self.lstm_delT_weights = nn.Parameter(lstm_del_he_factor * torch.randn(self.n_features,3*self.hidden_size,1,**factory_args))
        self.lstm_delT_bias = nn.Parameter(torch.zeros(self.n_features,3*self.hidden_size,**factory_args))
        
        self.lstm_c_inp_weights = nn.Parameter(torch.randn(self.n_features,self.hidden_size))
        self.lstm_c_out_weights = nn.Parameter(torch.randn(self.n_features,self.hidden_size))
        #Prediction network
        if self.memory_type=='Both':
            in_for_out_dim =self.hidden_size + self.hidden_size
        elif self.memory_type in ['LTM','STM']:
            in_for_out_dim = self.hidden_size
        else:
            in_for_out_dim = 0
            exit()
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
    
    def forward(self,tim,X,X_hap,mask,Ht,Ct):
        #tim is the instance no.
        #list of STM of each feature for aggregation
        H_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        # C_curr = torch.zeros(self.n_features,self.hidden_size,dtype=torch.float32,device=self.device)
        # Feature information
        self.feat_indices_curr = torch.arange(self.n_features).to(self.device)[mask==1]
        self.feat_indices_new = torch.arange(self.n_features).to(self.device)[mask&(~self.feat_observed)]
        self.feat_indices_old = torch.arange(self.n_features).to(self.device)[mask&self.feat_observed]
        self.feat_count[self.feat_indices_curr]+=1
        self.feat_observed = self.feat_observed | mask
        self.feat_indices_observed = torch.arange(self.n_features).to(self.device)[self.feat_observed==1]
        
        # Normalization of Input Data
        X_hap_t = self.normalize(X,mask,tim)
        #LSTM BLOCK
        delta = (tim - self.last_occured).to(device=self.device)
        #last_occured - latest time instance they appeared - if mask=1 tim else self.last_occured
        self.last_occured = tim*mask + (1^mask)*self.last_occured
        #Concat Input, Short term memory
        cur_input = torch.cat([X_hap_t.unsqueeze(-1), Ht[self.feat_indices_curr,:]], dim=-1)
        # Apply Constraint on time weights for T1 gate
        with torch.no_grad():
            self.lstm_delT_weights[:,self.hidden_size,:] = torch.clip(self.lstm_delT_weights[:,self.hidden_size,:],max=self.boundary)
        assert torch.sum(self.lstm_delT_weights[:,self.hidden_size,:]>self.boundary)<=0.0
        
        ## Forward Pass inside the TimeLSTM-3 Block
        cur_output = torch.matmul(self.lstm_weights[self.feat_indices_curr], cur_input.unsqueeze(-1).float()).squeeze(-1)
        xm_T_out = torch.matmul(self.lstm_xT_weights[self.feat_indices_curr],X_hap_t.unsqueeze(-1).unsqueeze(-1).float()).squeeze(-1)
        del_T_out = torch.matmul(self.lstm_delT_weights[self.feat_indices_curr],delta[self.feat_indices_curr].unsqueeze(-1).unsqueeze(-1).float()).squeeze(-1)
        if self.bias:
                cur_output = cur_output + self.lstm_bias[self.feat_indices_curr]
                xm_T_out = xm_T_out + self.lstm_xT_bias[self.feat_indices_curr]
        gate_input, gate_output, gate_pre_c = cur_output.chunk(3, dim=-1)
        xm_T1, xm_T2 = xm_T_out.chunk(2,dim=-1)
        del_T1,del_T2,del_To = del_T_out.chunk(3,dim=-1)
        del_T1 = torch.sigmoid(del_T1) # can also use tanh here
        del_T2 = torch.sigmoid(del_T2)
        T_1t = torch.sigmoid(xm_T1+del_T1)
        T_2t = torch.sigmoid(xm_T2+del_T2)
        
        gate_input = torch.sigmoid(gate_input+self.lstm_c_inp_weights[self.feat_indices_curr]*self.c_t)
        gate_pre_c = torch.tanh(gate_pre_c)
        agg_c_t_tilde = (1-gate_input*T_1t) * self.c_t + gate_input * T_1t * gate_pre_c
        # Long-term memory
        agg_c_t = (1-gate_input) * self.c_t + gate_input * T_2t * gate_pre_c
        gate_output = torch.sigmoid(gate_output+self.lstm_c_out_weights[self.feat_indices_curr]*agg_c_t_tilde+ del_To )
        
        ### Prediction Stream
        # Short-term Memory with zeros for unavailabe features
        H_curr[self.feat_indices_curr,:] = gate_output * torch.tanh(agg_c_t_tilde)
        # Aggregated long-term memory of only current feature space
        C_curr = self.agg_func(agg_c_t, self.aggregate_by)
        # Populate short-term memory with previous short-term memories, wherever unavaialble features
        H_curr = torch.where(mask.any().unsqueeze(-1), H_curr, Ht)
        if self.memory_type == 'Both':
            if self.feature_space=='Current':
                input_for_out = torch.reshape(torch.cat([C_curr,self.agg_func(H_curr[self.feat_indices_curr,:], self.aggregate_by)]),[-1])
            elif self.feature_space=='Universal':
                input_for_out = torch.reshape(torch.cat([C_curr,self.agg_func(H_curr[self.feat_indices_observed,:], self.aggregate_by)]),[-1])
        elif self.memory_type == 'LTM':
            if self.feature_space=='Current':
                input_for_out = self.agg_func(H_curr[self.feat_indices_curr,:], self.aggregate_by)
            elif self.feature_space=='Universal':
                input_for_out = self.agg_func(H_curr[self.feat_indices_observed,:], self.aggregate_by)
        else:
            input_for_out = C_curr
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
        ## Loop over all Time Instances
        for t in tqdm(range(X.shape[0])):
            optimizer.zero_grad()
            # forward pass
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
        ## Online Normalization
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