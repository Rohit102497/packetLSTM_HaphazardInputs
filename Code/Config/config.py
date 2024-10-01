# lr - learning rate
# hidden_size - 64
# bias - True, False
# aggregate_by - "Max", "Min", "Sum", "Mean" 
# decay - 'TimeLSTM_3', 'TimeLSTM_2', 'TimeLSTM_1', 'Decay', 'None'
# features_space - "Current", "Universal"
# relu_in_prediction - True, False
# normalization - 'None', "Z_score", 'Min_Max', 'Dec_Scal', 'Unit_Vec', 'Mean_Norm'

def magic04_p25_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def magic04_p50_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},        
    }
    return exp_no_dict[str(exp_no)]

def magic04_p75_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
     }
    return exp_no_dict[str(exp_no)]

def a8a_p25_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
        ##################################Memory ##############################################3
        "115":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},   

          #############################Scalability experiment################################################
        "10001":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},     
    }
    return exp_no_dict[str(exp_no)]

def a8a_p75_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
        #########################Memory Type#####################################################
        "115":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def a8a_p50_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0009,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
     }
    return exp_no_dict[str(exp_no)]

def imdb_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', "model_type":"LSTM"},
        "116":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', "model_type":"LSTM"},
        ###########################Model Arch###########################################################
        "117":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},        
    }
    return exp_no_dict[str(exp_no)]

def higgs__p50_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def higgs__p25_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def higgs__p75_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]  

def susy_p25_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def susy_p50_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def susy_p75_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ##############################Aggregation Module#########################################
        "102":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Mean","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "103":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Sum","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "104":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Min","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        #############################Time Modelling###############################################
        "105":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},    
        "106":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'Decay',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "107":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_2',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        "108":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_1',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ###########################Universal Features############################################
        "109":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Universal","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ############################Normalization###############################################
        "110":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Min_Max", 'model_type': 'LSTM'},
        "111":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Dec_Scal", 'model_type': 'LSTM'},
        "112":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Mean_Norm", 'model_type': 'LSTM'},
        "113":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Unit_Vec", 'model_type': 'LSTM'},
        "114":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"None", 'model_type': 'LSTM'},
         #########################Memory Type#####################################################
        "115":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'LTM', 'model_type': 'LSTM'},
        "116":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","memory_type":'STM', 'model_type': 'LSTM'},
         #########################RNN, GRU, LSTM#####################################################
        "117":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"RNN"},
        "118":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'None',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score","model_type":"GRU"},
    }
    return exp_no_dict[str(exp_no)]

def higgs_config(exp_no):
     exp_no_dict = {
          #########################Sudden#####################################################
          "201":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          #########################Obsolete#####################################################
          "202":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          #########################Reappearing#####################################################
          "203":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          #########################packetLSTM-Retraining at Each Interval#####################################################
          "301":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "302":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "303":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "304":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "305":{"lr":0.0002,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
     }
     return exp_no_dict[str(exp_no)]

def susy_config(exp_no):
     exp_no_dict = {
          #########################Sudden#####################################################
          "201":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          #########################Obsolete#####################################################
          "202":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          #########################Reappearing#####################################################
          "203":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          #########################packetLSTM-Retraining at Each Interval#####################################################
          "301":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "302":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "303":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "304":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
          "305":{"lr":0.0008,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
     }
     return exp_no_dict[str(exp_no)]

def data_config_other_p_diff_agg(data_name, exp_no):
     lr = {'magic04': 0.0006, 'a8a': 0.0009, 'susy': 0.0008, 'higgs': 0.0002}
     conf = {"hidden_size":64,"bias":True,"decay":'TimeLSTM_3', "feature_space":"Current",
             "relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM', "aggregate_by":"Max"}
     exp_no_dict = {
          #############################Benchmarking################################################
          "1001":{"aggregate_by":"Max"},
          "1002":{"aggregate_by":"Min"},
          "1003":{"aggregate_by":"Sum"},
          "1004":{"aggregate_by":"Mean"},
     }
     if exp_no in [1101 , 1102]:
          if data_name == "a8a":
               if exp_no == 1102:
                    conf["feature_space"] =  "Universal"
     else:          
          conf["aggregate_by"] = exp_no_dict[str(exp_no)]["aggregate_by"]
     conf["lr"] = lr[data_name]
     return conf

def data_config_alternating_variable_p(data_name, exp_no):
     lr = {'magic04': 0.0006, 'a8a': 0.0009, 'susy': 0.0008, 'higgs': 0.0002}
     conf = {"hidden_size":64,"bias":True,"decay":'TimeLSTM_3', "feature_space":"Current",
             "relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'}
     exp_no_dict = {
          #############################Benchmarking################################################
          "1001":{"aggregate_by":"Sum"},
          "1002":{"aggregate_by":"Mean"},
          "1003":{"aggregate_by":"Max"},
          "1004":{"aggregate_by":"Max"},
     }
     if exp_no == 1004:
          conf["feature_space"] =  "Universal"
     conf["aggregate_by"] = exp_no_dict[str(exp_no)]["aggregate_by"]
     conf["lr"] = lr[data_name]
     return conf

def get_config(data_name,prob,exp_no, syn_data_type):
     if data_name=='magic04':
          if syn_data_type == 'variable_p':
               if prob==0.25:
                    return magic04_p25_config(exp_no)
               elif prob==0.5:
                    return magic04_p50_config(exp_no)
               elif prob==0.75:
                    return magic04_p75_config(exp_no)
               else:
                    return data_config_other_p_diff_agg(data_name, exp_no)
          elif syn_data_type == "alternating_variable_p":
               return data_config_alternating_variable_p(data_name, exp_no)
     elif data_name=='a8a':
          if syn_data_type == 'variable_p':
               if prob==0.25:
                    return a8a_p25_config(exp_no)
               elif prob==0.5:
                    return a8a_p50_config(exp_no)
               elif prob==0.75:
                    return a8a_p75_config(exp_no) 
               else:
                    return data_config_other_p_diff_agg(data_name, exp_no)    
          elif syn_data_type == "alternating_variable_p":
               return data_config_alternating_variable_p(data_name, exp_no)
     elif data_name=='susy':
          if syn_data_type == 'variable_p':
               if prob==0.25:
                    return susy_p25_config(exp_no)
               elif prob==0.5:
                    return susy_p50_config(exp_no)
               elif prob==0.75:
                    return susy_p75_config(exp_no)
               else:
                    return data_config_other_p_diff_agg(data_name, exp_no)
          elif syn_data_type in ['sudden', 'obsolete', 'reappearing']:
               return susy_config(exp_no)
          elif syn_data_type == "alternating_variable_p":
               return data_config_alternating_variable_p(data_name, exp_no)
     elif data_name=='higgs':
          if syn_data_type == 'variable_p':
               if prob==0.5:
                    return higgs__p50_config(exp_no)
               elif prob==0.25:
                    return higgs__p25_config(exp_no)
               elif prob==0.75:
                    return higgs__p75_config(exp_no)
               else:
                    return data_config_other_p_diff_agg(data_name, exp_no)
          elif syn_data_type in ['sudden', 'obsolete', 'reappearing']:
               return higgs_config(exp_no)
          elif syn_data_type == "alternating_variable_p":
              return data_config_alternating_variable_p(data_name, exp_no)
     elif data_name=='imdb':
        return imdb_config(exp_no)