def get_config(data_name, exp_no):
    lr = {'magic04': 0.0002, 'a8a': 0.0002, 'susy': 0.0001, 'higgs': 0.0001, 'imdb': 0.0001}
    conf = {"hidden_size":32, "relu_in_prediction":True, "normalization":"Z_score", 
            "n_layers":1, "n_heads":2, "dropout":0.0}
    hidden_size = {'magic04': 128, 'a8a': 32, 'susy': 32, 'higgs': 32, 'imdb': 32}
    n_heads = {'magic04': 8, 'a8a': 16, 'susy': 4, 'higgs': 16, 'imdb': 4}
    # n_layers = {'magic04': 1, 'a8a': 16, 'susy': 2, 'higgs': 2, 'imdb': 4}
    exp_no_dict = {
        #############################Benchmarking################################################
        "0": {"hidden_size":32},
        "1": {"hidden_size":64},
        "2": {"hidden_size":128},
        "3": {"hidden_size":256},
        "4": {"hidden_size":512},

        "5": {"n_heads":1},
        "6": {"n_heads":4},
        "7": {"n_heads":8},
        "8": {"n_heads":16},

        "9": {"n_layers": 2},
        "10": {"n_layers": 3},
        "11": {"n_layers": 4},

        "12": {"lr": 0.001},
        "13": {"lr": 0.0005},
        "14": {"lr": 0.0001},
        "15": {"lr": 0.00005},

        "16": {"lr": 0.0003},
        "17": {"lr": 0.0002},
        "18": {"lr": 0.00009},
        "19": {"lr": 0.00008},
    }
    conf["lr"] = lr[data_name]

    if exp_no in [0, 1, 2, 3, 4]:
        conf["hidden_size"] = exp_no_dict[str(exp_no)]["hidden_size"]
    else:
        conf['hidden_size'] = hidden_size[data_name]
    
    if exp_no in [5, 6, 7, 8]:
        conf["n_heads"] = exp_no_dict[str(exp_no)]["n_heads"]
    else:
        conf['n_heads'] = n_heads[data_name]

    if exp_no in [9, 10, 11]:
        conf["n_layers"] = exp_no_dict[str(exp_no)]["n_layers"]

    if exp_no in [12, 13, 14, 15, 16, 17, 18, 19]:
        conf["lr"] = exp_no_dict[str(exp_no)]["lr"]
    return conf
