import pickle as pkl
import argparse
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='magic04',type=str,help="Name of the dataset")
parser.add_argument('--syn_data_type',default='sudden',type=str,choices=['sudden', 'obsolete', 'reappearing'], help='Class of Synthetic Data')
parser.add_argument('--exp_num',default='1',type=int,help='Experiment Number')
parser.add_argument('--nruns',type=int,default=5,help="Number of runs")
parser.add_argument('--methodname', default = "nb3", type = str,
                        choices = ["nb3", "fae", "olvf", "ocds", "ovfm", 
                                   "dynfo", "orf3v", "auxnet", "auxdrop", "olifl"],
                        help = "The name of the method")
args = parser.parse_args()
nruns = args.nruns

if args.methodname in ['olvf', 'olifl']:
    nruns = 1

path = os.path.realpath(__file__) 
result_addr = os.path.dirname(path) 
file_path = f'{result_addr}/Results/noassumption/{args.methodname}/{args.dataset}/Experiment_{args.exp_num}.data'

f = open(file_path,'rb')
data = pkl.load(f) 
result_dict = data['results']
keys = list(result_dict.keys())
result_final = result_dict[keys[0]]

o = {'labels': result_dict['labels'], 
        'predictions': result_dict['predictions']}

# Working with labels and predictions
if len(keys) > 1:
    number_of_instances = len(o['labels'])
    num_chunks = 5
    results = np.zeros((num_chunks, nruns))
    for n in range(nruns):
        # print("Run Number: ", n)
            instance_chunk = int(number_of_instances/num_chunks)
            i = 0
            for i in range(num_chunks-1):
                results[i, n] = balanced_accuracy_score(o['labels'][instance_chunk*i:instance_chunk*(i+1)], 
                                        o['predictions'][n][instance_chunk*i:instance_chunk*(i+1)])*100
            i = i+1
            results[i, n] = balanced_accuracy_score(o['labels'][instance_chunk*(i):], 
                                        o['predictions'][n][instance_chunk*(i):])*100
            

    mean = np.round(np.mean(results, axis = 1), 2)
    std = np.round(np.std(results, axis = 1), 2)

interval_list = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
for i in range(len(interval_list)):
    if args.methodname in ['olvf', 'olifl']:
        print(interval_list[i], ": ", mean[i])         
    else: 
        print(interval_list[i], ": ", mean[i], '+/-', std[i])


    

        