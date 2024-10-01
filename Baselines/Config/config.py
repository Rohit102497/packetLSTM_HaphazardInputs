# Model Configs
import numpy as np
from Utils.utils import dummy_feat, impute_data


# --------- NB3 ------------
def config_nb3(data_name):
    '''
        numTopFeats_percent = [.2, .4, .6, .8, 1]
    '''
    # config_dict = {}
    # config_dict["numTopFeats_percent"] = numTopFeats_percent

    params_list = {
        'magic04':      {"numTopFeats_percent":[0.6]},
        'a8a':          {"numTopFeats_percent":[0.2]},
        'susy':         {"numTopFeats_percent":[1]},
        'higgs':        {"numTopFeats_percent":[0.2]},
        'imdb':         {"numTopFeats_percent":[0.4]},
    }
    
    n_runs = 1 # NB3 is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
    config_dict = params_list[data_name]
    return n_runs, config_dict

# --------- FAE ------------
def config_fae(data_name):
    # Based on original paper
    config_dict = {}
    n_runs = 1 # FAE is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
    m = 5    # (maturity) Number of instances needed before a learner’s classifications are used by the ensemble
    p = 3    # (probation time) is the number of times in a row a learner is allowed to be under the threshold before being removed
    f = 0.15 # (feature change threshold) is the threshold placed on the amount of change between the
            # youngest learner’s set of features (yfs) and the top M features (mfs);
    r = 10   # (growth rate) is the number of instances between when the last learner was added and
            # when the ensemble’s accuracy is checked for the addition of a new learner
    N = 50   # Number of instances over which to compute an accuracy measure;
    params_list = {
        'magic04':      {"numTopFeats_percent":[1]},
        'a8a':          {"numTopFeats_percent":[0.2]},
        'susy':         {"numTopFeats_percent":[0.6]},
        'higgs':        {"numTopFeats_percent":[0.4]},
        'imdb':         {"numTopFeats_percent":[0.8]},
    }
    # Store all the config parameters
    config_dict["m"] = m
    config_dict["p"] = p
    config_dict["f"] = f
    config_dict["r"] = r
    config_dict["N"] = N
    config_dict["M"] = params_list[data_name]["numTopFeats_percent"]

    return n_runs, config_dict

# --------- OLVF ------------
def config_olvf(data_name, num_feat):
    n_runs = 1 # All w is 0. So it is deterministic
    
    params_list = {
        'magic04':      {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'imdb':         {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'a8a':          {'B':[1],       'C':[1],           'C_bar':[0.0001],   'reg':[0.0001]},
        'susy':         {'B':[1],       'C':[0.01],        'C_bar':[0.01],     'reg':[0.0001]},
        'higgs':        {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
    }
    
    config_dict = params_list[data_name]
    config_dict['n_feat0'] = num_feat
    
    return n_runs, config_dict

def config_ocds(num_feat, data_name):
    config_dict = {}
    gamma = [np.round(150/num_feat, 3)]# It is based on the number of features. The rule is to keep U_t < 150.
    if gamma[0] > 1:
        gamma = [1] # gamma cannot be more than 1

    params_list = {
        'magic04':      {'T':[16], 'alpha':[0.01], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.0001]},    
        'imdb':         {'T':[16], 'alpha':[1], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.01], 'beta2':[0.01]},  # Set heuristically
        'a8a':          {'T':[16], 'alpha':[1], 'gamma':gamma, 'beta0':[1], 
                        'beta1': [0.0001], 'beta2':[0.0001]},
        'susy':         {'T':[16], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.01], 
                        'beta1': [0.0001], 'beta2':[0.0001]}, # Set heuristically
        'higgs':        {'T':[8], 'alpha':[0.0001], 'gamma':gamma, 'beta0':[0.0001], 
                        'beta1': [0.01], 'beta2':[0.0001]}, # Set heuristically
    }
    config_dict = params_list[data_name]
    
    return config_dict

# --------- OVFM ------------
def config_ovfm(data_name):
    config_dict = {}
    '''

    'decay_choice': Possible values - 0, 1, 2, 3, 4

    'contribute_error_rate': 

    'decay_coef_change': This is used to update decay coefficient (this_decay_coef). Set as False
    
    'batch_size_denominator': This is used to update decay coefficient (this_decay_coef). 
                              If 'decay_coef_change' is False, then the value of 'batch_size_denominator'
                              does not matter.
    
    Below the hyperparameters corresponding to different datasets are defined.


        Heuristically Chosen: The following datasets requires significant time to run. Therefore,
                              we heuristically chose the hyperparameters based on the best performance
                              parameters of other similar size dataset.
                              ["imdb", "a8a", "susy", "higgs", ] 
        
        Hyperparameters Exhaustive Searching: Best hyperparameters for the following dataset were 
                              exhaustively search.
                              ["magic04"]
    '''
    params_dict = {
        "magic04"       : {'decay_choice': [3], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "imdb"          : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "a8a"           : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "susy"          : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "higgs"         : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
    }
    config_dict = params_dict[data_name]
    
    return config_dict
    
# --------- DynFo ------------
def config_dynfo(num_of_instances, data_name):
    ''' Dynfo takes a lot of time to run, because at each instance, the model undergoes many 
    relearning operations. To make sure, that the model does not undergo many relearning 
    operation, we need to set higher beta and theta1 values.
    '''
    config_dict = {}
    # Setting the value of N as 10% of the data or 20 instances. Whichever is less
    N = int(num_of_instances*.1)
    if N > 20:
        N = 20

    ''' Original paper provides the best hyperparameter for imdb dataset. We only change 
        values of beta and theta1 such that it is feasible to run the experiment.'''
    params_list = {
        "magic04":     {"alpha": [0.5], "beta": [0.5], "delta": [0.1], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "imdb":        {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "a8a":         {"alpha": [0.5], "beta": [0.5], "delta": [0.03], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "susy":        {"alpha": [0.5], "beta": [0.5], "delta": [0.4], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "higgs":       {"alpha": [0.5], "beta": [0.5], "delta": [0.2], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
                   
    }
    config_dict = params_list[data_name]

    return config_dict

# --------- ORF3V ------------
def config_orf3v(data_name):
        
    params_list = {
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    config_dict = params_list[data_name]

    return config_dict

# --------- Aux-Net ------------
def config_auxnet(data_name):
    config_dict = {}

    params_list = {
        "magic04":     {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.5]},
        "imdb":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "a8a":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "susy":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
        "higgs":       {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
    }
    
    config_dict = params_list[data_name]

    return config_dict


# --------- Aux-Drop ------------
def config_auxdrop(if_auxdrop_no_assumption_arch_change, X, data_name,
                    if_imputation, if_dummy_feat, n_dummy_feat, X_haphazard,
                    mask, imputation_type, dummy_type):
    config_dict = {}
    n_neuron_aux_layer_dict = {"magic04": 100, "susy": 100, "higgs": 100,
                              "a8a": 500, "imdb": 30000}
    max_num_hidden_layers = [6] # Number of hidden layers
    qtd_neuron_per_hidden_layer = [50] # Number of nodes in each hidden layer except the AuxLayer
    n_classes = 2 # The total number of classes (output labels)
    batch_size = 1 # The batch size is always 1 since it is based on stochastic gradient descent
    b = [0.99] # discount rate
    s = [0.2] # smoothing parameter

    n = {"magic04": [0.01],
         "a8a": [0.01], "susy": [0.05], "higgs": [0.05], 
         "imdb": [0.01]}
    
    dropout_p = {"magic04": [0.3],
         "a8a": [0.3], "susy": [0.3], "higgs": [0.3], 
         "imdb": [0.3]}

    n_aux_feat = X.shape[1] # Number of auxiliary features
    n_neuron_aux_layer = n_neuron_aux_layer_dict[data_name] # The total numebr of neurons in the AuxLayer
    use_cuda = False
    config_dict["max_num_hidden_layers"] = max_num_hidden_layers
    config_dict["qtd_neuron_per_hidden_layer"] = qtd_neuron_per_hidden_layer
    config_dict["n_classes"] = n_classes
    config_dict["n_neuron_aux_layer"] = n_neuron_aux_layer
    config_dict["batch_size"] = batch_size
    config_dict["b"] = b
    config_dict["s"] = s
    config_dict["n"] = n[data_name]
    config_dict["dropout_p"] = dropout_p[data_name]
    config_dict["n_aux_feat"] = n_aux_feat
    config_dict["use_cuda"] = use_cuda
    if if_auxdrop_no_assumption_arch_change:
        return config_dict
    else:
        # features_size - Number of base features
        # aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1. 
        features_size =  2 # number of base features
        aux_layer = [3] # The position of auxiliary layer. This code does not work if the AuxLayer position is 1.
        if if_imputation:
            n_aux_feat = X.shape[1] - features_size # We impute some features (feature_size) to create base features. 
            # Therefore number of base features would be total number of features - total number of base features
            # Create dataset
            X_base = impute_data(X_haphazard[:, :features_size],
                                mask[:, :features_size], imputation_type)
            X_aux_new = X_haphazard[:, features_size:]
            aux_mask = mask[:, features_size:]
        elif if_dummy_feat:
            features_size = n_dummy_feat # We create dummy feature as base feature
            # Create dataset
            X_base = dummy_feat(X_haphazard.shape[0], features_size, dummy_type)
            X_aux_new = X_haphazard
            aux_mask = mask
        config_dict["features_size"] = features_size
        config_dict["aux_layer"] = aux_layer
        config_dict["n_aux_feat"] = n_aux_feat
        return config_dict, X_base, X_aux_new, aux_mask
def config_olifl(data_name):
    
    params_list = {
        'magic04':      {'C':[1],             'option':[0],}, 
        'imdb':         {'C':[7e-3],          'option':[1], },
        'a8a':          {'C':[2],             'option':[1], },
        'susy':         {'C':[0.3],           'option':[1], },
        'higgs':        {'C':[8e-2],          'option':[1], },
        
    }
    config_dict = params_list[data_name]
    
    return config_dict