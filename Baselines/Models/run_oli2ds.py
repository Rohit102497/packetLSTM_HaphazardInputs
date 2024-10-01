from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from Models.OLIFL import OLIFL
from tqdm import tqdm
import numpy as np
import time

def create_param_list(model_params):
    
    params_list = []
    for C in model_params['C']:
        for lam in model_params['Lambda']:
            for B in model_params['B']:
                for theta in model_params['Theta']:
                    for sparse in model_params['Sparse']:
                        params_list.append({'C':C, 'Lambda':lam, 'B':B, 'theta':theta, 'sparse':sparse})

    return params_list

def run_OLI2DS(X, Y, X_haphazard, mask, num_runs, model_params):
    """
    X - 
    """
    print('Preparing Data to be fed')
    X_dict = []
    for i in tqdm(range(0, len(Y))):
        x, x_mask, y = X_haphazard[i], mask[i], Y[i]
        dct = {}
        for j in range(0,len(x)):
            if x_mask[j]==1:
                dct[j] = x[j]
        X_dict.append(dct)
        
    result = {}
    params_list = create_param_list(model_params)
    print("number of runs:", num_runs)
    for k in range(len(params_list)): # Different combination of params
        params = params_list[k]
        eval_list = []
        dict_pred = {}
        print(params)
        print('----------')
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = []
            Y_logits = []

            start_time = time.time()
            model = OLIFL(X_dict, Y, params['C'], params['Lambda'], params['B'], params['theta'], params['sparse'])
            Y_pred, Y_logits = model.fit(X, Y, X_haphazard, mask)
            taken_time = time.time() - start_time
            del model
            dict_pred[j] = Y_pred
            """for y in Y_logits:
                print(type(y),end=' ')"""
            eval_list.append(get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time))
        result[str(params)] = eval_list
        for p in eval_list:
            print(p)
        print('------')
        result['labels'] = Y
        result['predictions'] = dict_pred
     # The structure of results: It is dictionary with key being the number of Top M features and value are the metrics.
    return result