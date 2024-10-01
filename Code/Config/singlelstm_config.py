def get_config(data_name, exp_no):
    lr = {'magic04': 0.001, 'a8a': 0.0006, 'susy': 0.0001, 'higgs': 0.0002, 'imdb': 0.0008}
    conf = {"hidden_size":32, "relu_in_prediction":True, "normalization":"Z_score", 
            "num_layers":1, "dropout":0.0}
    conf["lr"] = lr[data_name]
    return conf
