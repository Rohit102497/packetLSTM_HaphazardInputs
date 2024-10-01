def get_config(data_name, exp_no):
    lr = {'magic04': 0.0002, 'a8a': 0.0002, 'susy': 0.0001, 'higgs': 0.0001, 'imdb': 0.0001}
    conf = {"hidden_size":32, "relu_in_prediction":True, "normalization":"Z_score", 
            "n_layers":1, "n_heads":2, "dropout":0.0}
    hidden_size = {'magic04': 128, 'a8a': 32, 'susy': 32, 'higgs': 32, 'imdb': 32}
    n_heads = {'magic04': 8, 'a8a': 16, 'susy': 4, 'higgs': 16, 'imdb': 4}
    conf["hidden_size"] = hidden_size[data_name]
    conf["lr"] = lr[data_name]
    conf["n_heads"] = n_heads[data_name]

    return conf
