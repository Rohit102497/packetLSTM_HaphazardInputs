# Libraries required
import numpy as np
import pandas as pd
import pickle
import os

def data_folder_path(data_folder, data_name):
    path = os.path.realpath(__file__) 
    dir = os.path.dirname(path) 
    storage_folder = dir.replace('Baselines/Data_Code', 'Datasets/')
    return storage_folder + data_folder + "/" + data_name

# Load magic04 data
def data_load_magic04(data_folder):
    data_name = "magic04.data"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[10] == 'g')*1
    data_initial = data_initial.iloc[:,:10]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load a8a data
def data_load_a8a(data_folder):
    data_name = "a8a.txt"
    n_feat = 123
    number_of_instances = 32561
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = " ", header = None, engine = 'python')
    data = pd.DataFrame(0, index=range(number_of_instances), columns = list(range(1, n_feat+1)))
    # 16th column contains only NaN value
    data_initial = data_initial.iloc[:, :15]
    for j in range(data_initial.shape[0]):
            l = [int(i.split(":")[0])-1 for i in list(data_initial.iloc[j, 1:]) if not pd.isnull(i)]
            data.iloc[j, l] = 1
    label = np.array(data_initial[0] == -1)*1
    data.insert(0, column='class', value=label)
    data = data.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load susy data
def data_load_susy(data_folder):
    data_name = "SUSY_1M.csv.gz"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, compression='gzip')
    label = np.array(data_initial["0"] == 1.0)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load susy data
def data_load_higgs(data_folder):
    data_name = "HIGGS_1M.csv.gz"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, compression='gzip')
    label = np.array(data_initial["0"] == 1.0)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

# Load imdb dataset:
def data_load_imdb(data_folder):
    data_name = "imdb"
    data_path = data_folder_path(data_folder, data_name)
    # To load the file
    with open(data_path, 'rb') as handle:
        data_initial = pickle.load(handle)

    data_initial = data_initial.astype(float)
    # Substitute each position containing -1 with nan value
    data_initial[data_initial == -1] = np.nan

    # Random shuffling of dataset
    np.random.shuffle(data_initial)

    # Rating <=4 is negative and >=7 is positive
    label = np.array(data_initial[:,0] >= 7)*1
    data_initial = data_initial[:,1:]
    
    Y = label.reshape(label.shape[0], 1)
    X = data_initial

    return X, Y