from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from Models.OLIFL import OLIFL
from tqdm import tqdm
import numpy as np
import time
def create_param_list(model_params):
    params_list = []
    for option in model_params['option']:
        if option==1:
            for C in model_params['C']:
                params_list.append({'C':C,'option':option})
        else:
            params_list.append({'C':1,'option':option})
    return params_list

def run_OLIFL(X, Y, X_haphazard, mask, num_runs, model_params):
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
            model = OLIFL(params['C'], params['option'],j)
            Y_pred, Y_logits = model.fit(X, Y, X_haphazard, mask)
            """for i in tqdm(range(0, len(Y))):
                x, x_mask, y = X_haphazard[i], mask[i], Y[i]
                y_pred, y_logit = model.partial_fit(x, x_mask, y)
                Y_pred.append(y_pred)
                Y_logits.append(y_logit)"""
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