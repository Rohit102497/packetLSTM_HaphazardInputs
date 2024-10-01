import pickle as pkl
import argparse
import numpy as np
import os
from Utils.utils import seed_everything, get_all_metrics
parser = argparse.ArgumentParser()

parser.add_argument('--dataset',default='magic04',choices=['magic04', 'a8a', 'susy', 'higgs', 'imdb'],type=str,help="Name of the dataset")
parser.add_argument('--availprob',default='0.9',type=float,help="Value of Probabilty")
parser.add_argument('--exp_num',default='1001',type=str,help='Experiment Number to print')
args = parser.parse_args()

args.dataset = 'kaggle_results/' + args.dataset
path = os.path.realpath(__file__) 
result_addr = os.path.dirname(path)

seed = 2024

logits_list = {}
labels_list = {}
prediction_list = {}
for i in range(5):
    file_path = f'{result_addr}/Results/{args.dataset}/p{int(100*args.availprob)}/Experiment_{args.exp_num}_{seed+i}.data'
            
    f = open(file_path,'rb')
    o = pkl.load(f)
    keys = list(o.keys())    
    result_dict = {}
    result_dict[keys[0]] = o[keys[0]]

    logits_list[i] = o['logits'][0]
    labels_list[i] = o['labels'].T[0]
    prediction_list[i] = o['predictions'][0]
    # print("keys: ", keys)
    # print(o['labels'])
    # print(o['predictions'])
    # print(o['logits'])

    # # print(o['labels'].T[0])
    # ninst = len(o['labels'])
    # print(ninst)
    # # res = get_all_metrics(Y.T[0],Y_pred,np.array(model.pred_logits),time_taken)
    # k = 5000
    # for i in range(5):
    #     for j in range(int(ninst/k) - 1):
    #         start, end = j*k, (j+1)*k
    #         res = get_all_metrics(o['labels'].T[0][start:end], o['predictions'][i][start:end], o['logits'][i][start:end])
    #         print(i, j, res['Bal. Accuracy'])
    #     start, end = (j+1)*k, ninst
    #     res = get_all_metrics(o['labels'].T[0][start:end], o['predictions'][i][start:end], o['logits'][i][start:end])
    #     print(i, j+1, res['Bal. Accuracy'])


    # # Printing the consolidated results:
    # dct = {}
    # for key in result_dict:
    #     dct[key] = []
    #     dct[key].append(result_dict[key]) 
    # exp_result = {}
    # print(f"Results of Experiment {args.exp_num} \n")
    # for key in dct:
    #     dct = dct[key][0]

    # metric_list = ['Errors', 'Accuracy', 'AUROC', 'AUPRC', 'Balanced Accuracy', 'Time (in sec)']
    # i = 0
    # for key in dct[0]:
    #     ls = [j[key] for j in dct]
    #     mean = np.mean(ls)
    #     std = np.std(ls)
    #     print(metric_list[i], ": ", mean, '+/-', std)
    #     i = i+1
ninst = 1000000
nblock = 10
block_size = int(ninst/nblock)
result = np.zeros((5, nblock))
for i in range(5):
    for j in range(nblock):
        res = get_all_metrics(labels_list[i][j*block_size:(j+1)*block_size], 
                             prediction_list[i][j*block_size:(j+1)*block_size], 
                             logits_list[i][j*block_size:(j+1)*block_size])
        result[i, j] = res['Bal. Accuracy']
print(np.mean(result, axis = 0))