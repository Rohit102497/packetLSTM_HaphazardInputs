import torch
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed")
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
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
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel

device = torch.device('cuda:' + str(0)) if torch.cuda.is_available() else torch.device('cpu')
#Code for Transformer
class online_nlp(nn.Module):
    def __init__(self,n_class:int,n_features:int,\
        device='cuda', exp_type:str=None, model_type:str=None,
        batch_size:int=1, 
        hidden_size:int=768, 
        normalization:str='Z_score',relu_in_prediction:bool=True,lr:float=None):
        super(online_nlp,self).__init__()
        """
        hidden_size- Output vector size of LSTM block
        
        """
        self.n_features = n_features
        self.n_class = n_class
        self.device = device

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.exp_type = exp_type
        self.model_type = model_type
        self.lr = lr

        print("Hidden Size: ", self.hidden_size)

        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased').to(device=device)
        elif self.model_type == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device=device)
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.normalization = 'Z_score'
        self.relu_in_prediction = relu_in_prediction
        self.bert_cls_size = 768
        in_for_out_dim = self.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.bert_cls_size,in_for_out_dim),
            nn.ReLU(),
            nn.Linear(in_for_out_dim, in_for_out_dim),
            nn.ReLU(),
            nn.Linear(in_for_out_dim, self.n_class),
        ).to(device=device)
        
        #Performance Evaluation
        self.prediction = []
        self.train_losses=[]
        self.pred_logits=[]
        if self.exp_type == 'InputPairs':
            self.mask_index = torch.arange(start = 1, end = self.n_features+1).to(device = self.device)
    
    def forward(self, X_hap, mask):

        if self.exp_type == 'OnlyValues':
            X_hap_str = ' '.join(map(str, X_hap[mask].tolist()))
        elif self.exp_type == 'InputPairs':
            '''
            Here we passed the normalized haphazard inputs without imputation. Only the non nan values are considered.
            We also pass the mask as feature index. For this the feature index is numbered from 1 till number of features.
            The non nan haphazard inputs and the available feature indices are concatenated in an alternate fashion (scissor way).
            Finally, the input is padded with zeros to match the length of 2*(number of features).
            Inputs looks like: [value1, f1, value2, f2, ..., pad] with shape 2*(number of features)
            '''
            emb_inp = torch.cat((X_hap[mask].unsqueeze(1), self.mask_index[mask].unsqueeze(1)), dim = 1) 
            X_hap_str = ' '.join(map(str, emb_inp.tolist()))
        # Convert X_hap to string
        
        # print(X_hap_str)
        # Tokenize the input
        inputs = self.tokenizer(X_hap_str, return_tensors="pt", padding='max_length', truncation=True, max_length=self.hidden_size)
        # print(inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Pass through BERT/DistilBERT
        # print(inputs['input_ids'].shape, inputs['attention_mask'].shape)
        outputs = self.model(**inputs)
        
        # Take the [CLS] token output
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Final prediction
        logits = self.mlp(cls_output)
        pred = F.softmax(logits, dim=-1)
        
        # Store predictions
        with torch.no_grad():
            self.prediction.append(torch.argmax(pred).detach().cpu().item())
            self.pred_logits.append(pred[0, 1].detach().cpu().numpy())
        
        return pred.squeeze(0)

    def fit(self,X,X_hap,Y,mask):
        
        self.prediction = []
        self.pred_logits=[]
        
        X=torch.tensor(X).to(self.device)
        X_hap=torch.tensor(X_hap).to(self.device)
        Y=torch.tensor(Y).to(self.device,dtype=torch.int)
        mask=torch.tensor(mask,dtype=torch.bool).to(self.device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss #().to(self.device)
        criterion_fn = criterion(reduction='none').to(self.device)
        
        for t in tqdm(range(X.shape[0])):
            optimizer.zero_grad()
            Y_predicted = self.forward(X_hap[t].float(), mask[t])
            loss = criterion_fn(Y_predicted.view(self.batch_size, self.n_class), Y[t].view(self.batch_size).long())            
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            # break
    


    