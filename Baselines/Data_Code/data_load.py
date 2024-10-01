# Libraries required
import numpy as np
from Data_Code.data_utils import data_load_magic04, data_load_a8a, data_load_susy, data_load_higgs
from Data_Code.data_utils import data_load_imdb

def seed_everything(seed: int):
	#random.seed(seed)
	#os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
"""	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True"""

seed_everything(42)

def check_mask_each_instance(mask):
    index_0 = np.where(np.sum(mask, axis = 1) == 0)
    random_index = np.random.randint(mask.shape[1], size = (len(index_0[0])))
    # print(mask.shape, index_0, len(index_0[0]))
    for i in range(len(index_0[0])):
        mask[index_0[0][i], random_index[i]] = 1
    return mask

def create_mask(data_folder, number_of_instances, n_feat, type, p_available):
    if type == "variable_p":
        mask = (np.random.random((number_of_instances, n_feat)) < p_available).astype(float)
    if type == "sudden":
        if data_folder == "higgs":
            num_chunks = 5
            instance_chunk = int(number_of_instances/num_chunks)
            mask = np.zeros((number_of_instances, n_feat))
            feat_chunk = int(n_feat/num_chunks)
            i = 0
            for i in range(num_chunks-1):
                mask[instance_chunk*i:instance_chunk*(i+1), :feat_chunk*(i+1)] = 1
            mask[instance_chunk*(i+1):, :] = 1
        elif data_folder == "susy":
            num_chunks = 5
            instance_chunk = int(number_of_instances/num_chunks)
            mask = np.zeros((number_of_instances, n_feat))
            feat_chunk = int(n_feat/num_chunks)
            i = 0
            for i in range(num_chunks-1):
                mask[instance_chunk*i:instance_chunk*(i+1), :int(np.round((n_feat/num_chunks)*(i+1)))] = 1
            mask[instance_chunk*(i+1):, :] = 1
    if type == "obsolete":
        if data_folder == "higgs":
            num_chunks = 5
            instance_chunk = int(number_of_instances/num_chunks)
            mask = np.zeros((number_of_instances, n_feat))
            feat_chunk = int(n_feat/num_chunks)
            i = 0
            for i in range(num_chunks-1):
                mask[instance_chunk*i:instance_chunk*(i+1), feat_chunk*i:] = 1
            mask[instance_chunk*(i+1):, feat_chunk*(i+1):] = 1
        elif data_folder == "susy":
            num_chunks = 5
            instance_chunk = int(number_of_instances/num_chunks)
            mask = np.zeros((number_of_instances, n_feat))
            feat_chunk = int(n_feat/num_chunks)
            i = 0
            for i in range(num_chunks-1):
                mask[instance_chunk*i:instance_chunk*(i+1), :int(np.round((n_feat/num_chunks)*(num_chunks-i)))] = 1
            mask[instance_chunk*(i+1):, :int(np.round((n_feat/num_chunks)*(num_chunks-i-1)))] = 1
    if type == "reappearing":
        instance_num_chunks = 5
        instance_chunk = int(number_of_instances/instance_num_chunks)
        feat_num_chunks = 2
        feat_chunk = int(n_feat/feat_num_chunks)
        mask = np.zeros((number_of_instances, n_feat))
        for i in range(instance_num_chunks):
            if i%2 == 0:
                start = 0
                end = feat_chunk
            else:
                start = feat_chunk
                end = n_feat
            mask[instance_chunk*i:instance_chunk*(i+1), start:end] = 1
    return mask

def data_load_synthetic(data_folder = "magic04", type = "variable_p", p_available = 0.75):

    if data_folder == "magic04":
        X, Y = data_load_magic04(data_folder)
    elif data_folder == "a8a":
        X, Y = data_load_a8a(data_folder)
    elif data_folder == "susy":
        X, Y = data_load_susy(data_folder)
    elif data_folder == "higgs":
        X, Y = data_load_higgs(data_folder)
    
    n_feat = X.shape[1]
    number_of_instances = X.shape[0]
    # create mask
    mask = create_mask(data_folder, number_of_instances, n_feat, type, p_available)
    mask = check_mask_each_instance(mask)
    X_haphazard = np.where(mask, X, 0)
    return X, Y, X_haphazard, mask
        
        
def data_load_real(data_folder = "imdb"):
    if data_folder == "imdb":
        X, Y = data_load_imdb(data_folder)    
        mask = np.ones((X.shape))
        mask[np.isnan(X)] = 0
        X_haphazard = np.where(mask, X, 0)
        return X, Y, X_haphazard, mask
    
    