def get_config(data_name, exp_no):
    lr = {'magic04': 0.00005, 'a8a': 0.00005, 'susy': 0.00005, 'higgs': 0.00005, 'imdb': 0.00005}
    conf = {"hidden_size":768, "relu_in_prediction":True, "normalization":"Z_score"}
    conf["lr"] = lr[data_name]
    return conf
