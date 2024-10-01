#---------IMPORT LIBRARIES--------------------
import numpy as np
import numpy.ma as ma
import torch
import os
import pickle
import time
import math
import os
import argparse
import sys

#------------------Import Code------------------
from Utils.utils import seed_everything, get_all_metrics
seed_everything(42)
from PacketTimeLSTM_1 import PacketTimeLSTM_1
from PacketTimeLSTM_2 import PacketTimeLSTM_2
from PacketTimeLSTM_3 import PacketTimeLSTM_3
from PacketLSTM_GRU_D import PacketLSTM_GRU_D
from PacketRNN import PacketRNN
from PacketGRU import PacketGRU
from Config.config import get_config
from Data_Code.data_load import data_load_synthetic, data_load_real

#--------------------DATA LOADING------------------
def load_data(data_name, syn_data_type='variable_p', p_available=None, feature_set = None, interval_set = None):
    if data_name in ["magic04", "a8a", "susy", "higgs"]:
        data_type = "Synthetic"
    else: # ["imdb"]
        data_type = "Real"

    #--------------Results Path--------------#
    path = os.path.realpath(__file__) 
    result_addr = os.path.dirname(path) 

    #--------------Load Data wrt Variables--------------#
    if data_type == "Synthetic":
        ''' 
            For all the experiments feature set and interval_set is None. 

            However, as done in the paper, we can pass feature set as 1 or 2, 
            where 1 represent the first 50% features (Feature 1 till 10 for higgs) and 
            2 means the next 50% features (Feature 11 till 21 for higgs).
            Same definition holds for SUSY.

            Interval_set can take values from 1 to 5, where interval 1 means the first 20% instances,
            interval 2 means the next 20% instances and so on.
        '''
        X, Y, X_haphazard, mask = data_load_synthetic(data_name, syn_data_type, p_available, feature_set, interval_set)
        result_addr = result_addr + '/Results/'+data_name+'/p'+str(int(p_available*100))+'/Experiment_'
    else:
        X, Y, X_haphazard, mask = data_load_real(data_name)
        result_addr = result_addr + '/Results/'+data_name+'/Experiment_'
    return X, X_haphazard, Y, mask, result_addr

#---------------------Run Model----------------------------------
def run_model(X, X_haphazard,mask,Y,num_runs, result_addr, experiment_no):
    result = {}
    params = get_config(args.dataset,args.availprob,experiment_no, args.syn_data_type)
    eval_list = []
    if params['model_type']=='LSTM':
        if params['decay']=='TimeLSTM_1':
            model_name =  PacketTimeLSTM_1
        elif params['decay']=='TimeLSTM_2':
            model_name = PacketTimeLSTM_2
        elif params['decay']=='TimeLSTM_3':
            model_name = PacketTimeLSTM_3
        elif params['decay']=='Decay' or params['decay']=='None':
            model_name = PacketLSTM_GRU_D
    elif params['model_type']=='RNN':
        model_name = PacketRNN
    elif params['model_type']=='GRU':
        model_name = PacketGRU
    else:
        print("Please select correct model ['LSTM','RNN','GRU']")
        exit()
    del params['model_type']
    print(f"Experiment No:- {experiment_no} \n Params: {params}")
    print(f"Number of runs:-{num_runs}")
    dict_pred = {}
    dict_logits = {}
    for j in range(num_runs):
        seed_everything(args.seed+j)
        start_time = time.time()
        model = model_name(n_class=len(np.unique(Y)),n_features=X.shape[1],device=device,**params)
        model.to(device)
        model.fit(X,X_haphazard,Y,mask)
        Y_pred = np.array(model.prediction)
        dict_pred[j] = Y_pred
        dict_logits[j] = np.array(model.pred_logits)
        time_taken = time.time()-start_time
        res = get_all_metrics(Y.T[0],Y_pred,np.array(model.pred_logits),time_taken)
        print(f"Run number {j+1} \n metrics are:-{res}")
        eval_list.append(res)
        del model
    result[str(params)] = eval_list
    result['labels'] = Y
    result['predictions'] = dict_pred
    result['logits'] =  dict_logits
    result_addr += f'{str(experiment_no)}.data'
    directory, filename = os.path.split(result_addr)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(result_addr, 'wb') as file: 
        pickle.dump(result, file) 
    return result

#---------------------Arguments----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nruns',type=int,default=5,help="No of seeds to run on")
    parser.add_argument('--seed',type=int,default=2024,help="Set seed")
    parser.add_argument('--exp_num', type=int,help="experiment number to run")
    parser.add_argument('--syn_data_type',default='variable_p',type=str,choices=['variable_p', 'sudden', 'obsolete', 'reappearing', 'alternating_variable_p'], 
                        help='Class of Synthetic Data. alternating_variable_p means alternative 100 instance have p_values 0.25 and 0.75')
    parser.add_argument('--dataset',default='magic04',choices=['magic04', 'a8a', 'susy', 'higgs', 'imdb'],type=str,help='The name of the dataset')
    parser.add_argument('--cuda', default=False, action=argparse.BooleanOptionalAction, help='True if using gpu. Default False')
    parser.add_argument('--availprob',default=0.75,type=float,help='Set the Probabilty')
    parser.add_argument('--gpu_no', default=0, type=int, help='The number of gpu to use. Default 0')
    parser.add_argument('--feature_set', default=None, type=int, choices=[1, 2], 
                        help='The feature set for higgs datset. 1 and 2 represent the first 10 and the last 11 features respectively.')
    parser.add_argument('--interval_set', default=None, type=int, choices=[1, 2, 3, 4, 5], 
                        help='1 represent first 200k instances, 2 represent the next 200k instances, and so on.')
    ''' 
            For all the experiments feature set and interval_set is None. 

            However, as done in the paper, we can pass feature set as 1 or 2, 
            where 1 represent the first 50% features (Feature 1 till 10 for higgs) and 
            2 means the next 50% features (Feature 11 till 21 for higgs).

            Same definition holds for SUSY.

            Interval_set can take values from 1 to 5, where interval 1 means the first 20% instances,
            interval 2 means the next 20% instances and so on.
    '''
    args = parser.parse_args()
    if args.cuda:
        device = torch.device('cuda:' + str(args.gpu_no)) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    X, X_haphazard, Y, mask,result_addr = load_data(args.dataset, args.syn_data_type, args.availprob, args.feature_set, args.interval_set)
    result = run_model(X, X_haphazard, mask, Y, args.nruns, result_addr, args.exp_num)
    

