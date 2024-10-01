def get_config(data_name, exp_no):
    lr = {'magic04': 0.0001, 'a8a': 0.0002, 'susy': 0.00008, 'higgs': 0.00006, 'imdb': 0.00007}
    conf = {"hidden_size":64, "num_inds":32, "n_heads":2,"normalization":"Z_score"}
    hidden_size = {'magic04': 64, 'a8a': 32, 'susy': 32, 'higgs': 32, 'imdb': 32}
    n_heads = {'magic04': 1, 'a8a': 2, 'susy': 8, 'higgs': 1, 'imdb': 2}
    num_inds = {'magic04': 64, 'a8a': 32, 'susy': 256, 'higgs': 64, 'imdb': 32}
    exp_no_dict = {
        #############################Hidden Size################################################
        "1": {"hidden_size":32},
        "2": {"hidden_size":64},
        "3": {"hidden_size":128},
        "4": {"hidden_size":256},
        "5": {"hidden_size":512},

        #############################Num Inds################################################
        "6": {"num_inds":64},
        "7": {"num_inds":128},
        "8": {"num_inds":256},
        "9": {"num_inds":512},

        #############################Num Heads################################################
        "10": {"n_heads":1},
        "11": {"n_heads":8},
        "12": {"n_heads":4},

        #############################Learning rate################################################
        "13": {"lr":0.001},
        "14": {"lr":0.0005},
        "15": {"lr":0.0001},
        "16": {"lr":0.00005},

        "17": {"lr":0.00006},
        "18": {"lr":0.00007},
        "19": {"lr":0.00008},
        "20": {"lr":0.00009},

        "31": {"lr":0.0004},
        "32": {"lr":0.0003},
        "33": {"lr":0.00009},
        "34": {"lr":0.00008},
        "35": {"lr":0.00007},
        "36": {"lr":0.00006},


        #############################Num Heads - HIGGS and SUSY################################################
        "21": {"n_heads":2},
        "22": {"n_heads":4},
        "23": {"n_heads":8},
        
        #############################Learning rate - HIGGS and SUSY################################################
        "51": {"lr":0.0002},
        "52": {"lr":0.0004},
        "53": {"lr":0.0006},
        "54": {"lr":0.0008},
        
    }
    conf["lr"] = lr[data_name]
    conf["hidden_size"] = hidden_size[data_name]
    conf["num_inds"] = num_inds[data_name]
    conf["n_heads"] = n_heads[data_name]

    if exp_no in [1, 2, 3, 4, 5]:
        print("Search Hidden Size")
        conf["hidden_size"] = exp_no_dict[str(int(exp_no))]["hidden_size"]
    if exp_no in [6, 7, 8, 9]:
        print("Search Num Inds")
        conf["num_inds"] = exp_no_dict[str(int(exp_no))]["num_inds"]
    if exp_no in [10, 11, 12]:
        print("Search Num Heads")
        conf["n_heads"] = exp_no_dict[str(int(exp_no))]["n_heads"]
    if exp_no in [13, 14, 15, 16, 17, 18, 19, 20, 31, 32, 33, 34, 35, 36]:
        print("Search Learning Rate")
        conf["lr"] = exp_no_dict[str(int(exp_no))]["lr"]
    if data_name in ['higgs', 'susy'] and exp_no in [21, 22, 23]:
        print("Search Num Heads")
        conf["n_heads"] = exp_no_dict[str(int(exp_no))]["n_heads"]
    # if data_name in ['higgs', 'susy'] and exp_no in [51, 52, 53, 54]:
    #     print("Search Learning Rate for higgs and susy")
    #     conf["lr"] = exp_no_dict[str(int(exp_no))]["lr"]

    return conf
