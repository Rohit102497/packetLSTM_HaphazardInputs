def get_config(data_name, exp_no):
    lr = {'magic04': 0.0005, 'a8a': 0.00009, 'susy': 0.00009, 'higgs': 0.00008, 'imdb': 0.00007}
    conf = {"hidden_size":64, "n_heads":2,"normalization":"Z_score"}
    hidden_size = {'magic04': 64, 'a8a': 256, 'susy': 32, 'higgs': 64, 'imdb': 512}
    n_heads = {'magic04': 2, 'a8a': 2, 'susy': 2, 'higgs': 4, 'imdb': 1}
    exp_no_dict = {
        #############################Hidden Size################################################
        "1": {"hidden_size":32},
        "2": {"hidden_size":64},
        "3": {"hidden_size":128},
        "4": {"hidden_size":256},
        "5": {"hidden_size":512},

        #############################Num Heads################################################
        "10": {"n_heads":1},
        "11": {"n_heads":8},
        "12": {"n_heads":4},

        #############################Learning rate################################################
        "13": {"lr":0.001},
        "14": {"lr":0.0005},
        "15": {"lr":0.0001},
        "16": {"lr":0.00005},

        "17": {"lr":0.0007},
        "18": {"lr":0.0006},
        "19": {"lr":0.0004},
        "20": {"lr":0.0003},

        "21": {"lr":0.0004},
        "22": {"lr":0.0003},

        "23": {"lr":0.00009},
        "24": {"lr":0.00008},
        "25": {"lr":0.00007},
        "26": {"lr":0.00006},
        "27": {"lr":0.00004},
        "28": {"lr":0.00003},
        "29": {"lr":0.00002},
        "30": {"lr":0.00001},
        
        #############################Learning rate - HIGGS and SUSY################################################
        "51": {"lr":0.0002},
        "52": {"lr":0.0004},
        "53": {"lr":0.0006},
        "54": {"lr":0.0008},
        
    }
    conf["lr"] = lr[data_name]
    conf["hidden_size"] = hidden_size[data_name]
    conf["n_heads"] = n_heads[data_name]

    if exp_no in [1, 2, 3, 4, 5]:
        print("Search Hidden Size")
        conf["hidden_size"] = exp_no_dict[str(int(exp_no))]["hidden_size"]
    if exp_no in [10, 11, 12]:
        print("Search Num Heads")
        conf["n_heads"] = exp_no_dict[str(int(exp_no))]["n_heads"]
    if exp_no in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        print("Search Learning Rate")
        conf["lr"] = exp_no_dict[str(int(exp_no))]["lr"]
    if data_name in ['higgs', 'susy', 'imdb'] and exp_no in [51, 52, 53, 54]:
        print("Search Learning Rate for higgs, susy and imdb")
        conf["lr"] = exp_no_dict[str(int(exp_no))]["lr"]

    return conf
