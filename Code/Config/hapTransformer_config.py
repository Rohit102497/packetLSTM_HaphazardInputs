def get_config(data_name, exp_no):
    lr = {'magic04': 0.0005, 'a8a': 0.00009, 'susy': 0.00009, 'higgs': 0.00008, 'imdb': 0.00007}
    conf = {"hidden_size":64, "n_heads":2,"normalization":"Z_score"}
    hidden_size = {'magic04': 64, 'a8a': 256, 'susy': 32, 'higgs': 64, 'imdb': 512}
    n_heads = {'magic04': 2, 'a8a': 2, 'susy': 2, 'higgs': 4, 'imdb': 1}
    conf["lr"] = lr[data_name]
    conf["hidden_size"] = hidden_size[data_name]
    conf["n_heads"] = n_heads[data_name]

    return conf
