def get_config(data_name, exp_no):
    lr = {'magic04': 0.0002, 'a8a': 0.0002, 'susy': 0.0001, 'higgs': 0.0001, 'imdb': 0.0001}
    conf = {"hidden_size":512, "relu_in_prediction":True, "normalization":"Z_score"}
    # hidden_size = {'magic04': 128, 'a8a': 32, 'susy': 32, 'higgs': 32, 'imdb': 32}
    exp_no_dict = {
        #############################Benchmarking################################################
        "1": {"lr": 0.001},
        "2": {"lr": 0.0005},
        "3": {"lr": 0.0001},
        "4": {"lr": 0.00005},
    }
    conf["lr"] = lr[data_name]
    if exp_no in [1, 2, 3, 4]:
        conf['lr'] = exp_no_dict[str(exp_no)]['lr']
    return conf
