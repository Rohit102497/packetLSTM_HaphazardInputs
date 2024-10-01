import pickle as pkl
import argparse
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='susy',choices=['susy', 'higgs'],type=str,help="Name of the dataset")
parser.add_argument('--syn_data_type',default='sudden',type=str,choices=['sudden', 'obsolete', 'reappearing'], help='Class of Synthetic Data')
parser.add_argument('--exp_num',default='1',type=int,help='Experiment Number')
parser.add_argument('--nruns',type=int,default=5,help="Number of runs")
args = parser.parse_args()

availprob = 0.75
path = os.path.realpath(__file__) 
result_addr = os.path.dirname(path) 
file_path = f'{result_addr}/Results/{args.dataset}/p{int(100*availprob)}/Experiment_{args.exp_num}.data'

f = open(file_path,'rb')
o = pkl.load(f)
keys = list(o.keys())    
result_dict = {}
result_dict[keys[0]] = o[keys[0]]

# Working with labels and predictions
nruns =  args.nruns
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
    print(interval_list[i], ": ", mean[i], '+/-', std[i])




    

        
