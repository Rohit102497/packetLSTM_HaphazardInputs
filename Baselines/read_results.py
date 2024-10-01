import argparse
import pickle
import pandas as pd
import os

path = os.path.realpath(__file__) 
result_addr = os.path.dirname(path) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default = "magic04", type = str,
                        choices = ["imdb", "higgs", "susy", "a8a", "magic04"],
                        help='The name of the data')
    parser.add_argument('--availprob', default = 0.75, type = float,
                        help = "The probability of each feature being available to create synthetic data")
    parser.add_argument('--methodname', default = "nb3", type = str,
                        choices = ["nb3", "fae", "olvf", "ocds", "ovfm", 
                                   "dynfo", "orf3v", "auxnet", "auxdrop","olifl"],
                        help = "The name of the method")
    parser.add_argument('--exp_num',default='101',type=int,help='Experiment Number to print')
    args = parser.parse_args()
    type = "noassumption"
    data_name = args.dataset
    p_available = args.availprob
    method_name = args.methodname

    data_name_list = []
    if data_name == "synthetic":
        data_name_list = ["magic04", "higgs", "susy", "a8a"]
    elif data_name == "real":
        data_name_list = ["imdb"] 
    else:
        data_name_list = [data_name]

    for data_name in data_name_list:

        result_addr = result_addr + "/Results/" + type + "/" + method_name + "/" + data_name

        data_type = "Synthetic"
        if data_name in ["imdb", "diabetes_us", "spamassasin", "naticusdroid", "crowdsense_c3", "crowdsense_c5"]:
            data_type = "Real"

        if data_type == "Synthetic":
            result_addr = result_addr + "_prob_" + str(int(p_available*100)) + "/Experiment_" + str(args.exp_num) + ".data"
        else:
            result_addr = result_addr + "/Experiment_" + str(args.exp_num) + ".data"

        file = open(result_addr, 'rb') 
        data = pickle.load(file) 

        i  =  list(data['results'].keys())[0]

        # Calculate mean 
        mean_values_list = []
        val_list = pd.DataFrame(data['results'][i]).mean(axis = 0).values.tolist()
        mean_values_list.append(['%.2f' % elem for elem in val_list])

        # Calculate std deviation 
        std_values_list = []
        val_list = pd.DataFrame(data['results'][i]).std(axis = 0).values.tolist()
        std_values_list.append(['%.2f' % elem for elem in val_list])

        metric_list = ['Errors', 'Accuracy', 'AUROC', 'AUPRC', 'Balanced Accuracy', 'Time (in sec)']
        i = 0
        for i in range(len(metric_list)):
            if method_name in ["nb3", "fae", "olvf","olifl"]:
                print(metric_list[i], ": ", mean_values_list[0][i])
            else:
                print(metric_list[i], ": ", mean_values_list[0][i], '+/-', std_values_list[0][i])
            i = i+1

