def get_config(data_name, exp_no):
    lr = {'magic04': 0.0001, 'a8a': 0.0002, 'susy': 0.00008, 'higgs': 0.00006, 'imdb': 0.00007}
    hidden_size = {'magic04': 64, 'a8a': 32, 'susy': 32, 'higgs': 32, 'imdb': 32}
    n_heads = {'magic04': 1, 'a8a': 2, 'susy': 8, 'higgs': 1, 'imdb': 2}
    num_inds = {'magic04': 64, 'a8a': 32, 'susy': 256, 'higgs': 64, 'imdb': 32}

    conf = {"hidden_size":64, "num_inds":32, "n_heads":2,"normalization":"Z_score"}
    
    conf["lr"] = lr[data_name]
    conf["hidden_size"] = hidden_size[data_name]
    conf["num_inds"] = num_inds[data_name]
    conf["n_heads"] = n_heads[data_name]

    return conf
