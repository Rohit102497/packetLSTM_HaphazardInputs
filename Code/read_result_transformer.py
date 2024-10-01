import pickle as pkl
import argparse
import numpy as np
import os
parser = argparse.ArgumentParser()

parser.add_argument('--dataset',default='magic04',choices=['magic04', 'a8a', 'susy', 'higgs', 'imdb'],type=str,help="Name of the dataset")
parser.add_argument('--availprob',default='0.75',type=float,help="Value of Probabilty")
parser.add_argument('--exp_num',default='101',type=int,help='Experiment Number to print')
parser.add_argument('--exp_type', required=True, type = str, help = 'Type of experiment run',
                    choices = ['InputPairs', "PaddedInputs"])
args = parser.parse_args()

path = os.path.realpath(__file__) 
result_addr = os.path.dirname(path) 

if args.dataset != 'imdb':
    result_addr = f'{result_addr}/Results/Transformer/{args.exp_type}/{args.dataset}/p{str(int(args.availprob*100))}/Experiment_{args.exp_num}.data' 
else:
    result_addr = f'{result_addr}/Results/Transformer/{args.exp_type}/{args.dataset}/Experiment_{args.exp_num}.data'
        
f = open(result_addr,'rb')
o = pkl.load(f)
keys = list(o.keys())    
result_dict = {}
result_dict[keys[0]] = o[keys[0]]

# Printing the consolidated results:
dct = {}
for key in result_dict:
    dct[key] = []
    dct[key].append(result_dict[key]) 
exp_result = {}
print(f"Results of Experiment {args.exp_num} \n")
for key in dct:
    dct = dct[key][0]

metric_list = ['Errors', 'Accuracy', 'AUROC', 'AUPRC', 'Balanced Accuracy', 'Time (in sec)']
i = 0
res_display = []
for key in dct[0]:
    ls = [j[key] for j in dct]
    mean = np.mean(ls)
    std = np.std(ls)
    print(metric_list[i], ": ", mean, '+/-', std)
    res_display.append(str(np.round(mean, 2)) + '+/-' + str(np.round(std, 2)))
    i = i+1
print(res_display)
            