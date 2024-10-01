# packetLSTM: Dynamic LSTM Framework for Streaming Data with Varying Feature Space

## Overview
This repository contains datasets and implementation codes of different models for the paper, titled "packetLSTM: Dynamic LSTM Framework for Streaming Data with Varying Feature Space".

## Requirements
1. packetLSTM: To run the packetLSTM code, install libraries provided in the Code folder by `pip install -r Code/requirements.txt`.
2. Baselines: To run the Baselines code, install libraries provided in the Baselines folder by `pip install -r Baselines/requirements.txt`.

## Running packetLSTM
Refer to the `Code` Folder.

### Control Parameters
The paramters of the model is defined in the `Code/Config/config.py`. To run any experiment, you need to set the parameters value there with an exp_num. The parameters are:
For **config.py**
1. `lr` - learning rate
2. `hidden_size` - hidden size of the RNN/GRU/LSTM block; default value 64
3. `bias` - Use Bias or not [True, False], Default - True
4. `model_type` - Which Model to use? ['LSTM', 'RNN', 'GRU'], Default - 'LSTM'
5. `aggregate_by` - Aggregation Operator to use ['Max', 'Min', 'Sum', 'Mean'], Default - 'Max'
6. `decay` - Time-Modeling method to use ['TimeLSTM_3', 'TimeLSTM_2', 'TimeLSTM_1', 'Decay', 'None'], Default - 'TimeLSTM_3'
7. `features_space` - Which feature space to use for prediction ["Current", "Universal"], Default- "Current"
8. `relu_in_prediction` - Use ReLU or not in Fully Connected Layer [True, False], Default- True
9. `normalization` - Normalization method to use ['None', "Z_score", 'Min_Max', 'Dec_Scal', 'Unit_Vec', 'Mean_Norm'], Default- 'Z-score'
10. `memory_type` - What memory to use? ['LTM','STM','Both'], Default- 'Both'

For all the experiments in the paper, you can find the parameters already set in the `Code/Config/config.py` file with corresponding experiment number and its description. For example, exp_num = 101 corresponds to the benchmarking results. exp_num = 102, 103, and 104 corresponds to the Ablation study of aggregator operator. 

An example for parameters in `Code/Config/config.py` file:
```
def magic04_p75_config(exp_no):
    exp_no_dict = {
        #############################Benchmarking################################################
        "101":{"lr":0.0006,"hidden_size":64,"bias":True,"aggregate_by":"Max","decay":'TimeLSTM_3',
             "feature_space":"Current","relu_in_prediction":True,"normalization":"Z_score", 'model_type': 'LSTM'},
        ...
        ...
    }
    return exp_no_dict[str(exp_no)]
```

### Running Code Parameters

To run packetLSTM, we need to pass below parameters with the file `Code/run_model.py`

1. `--nruns` - Number to times to run the model with different seeds. Default = 5 for packetLSTM.
2. `--seed` - Initial seed ,increments by every run, i.e seed,seed+1,..,seed+nruns
3. `--exp_num` - Experiment Number set in `Config/config.py`
4. `--syn_data_type` - Class of synthetic data ['variable_p', 'sudden', 'obsolete', 'reappearing']
5. `--dataset` - dataset to use ['magic04','a8a','imdb','susy','higgs']
6. `--availprob` - Probabilty of synthetic data feature appearing in an instance. In our paper we used [0.25,0.5,0.75]
7. `--feature_set` - 'The feature set for higgs and susy datset for packetLSTM-retraining expriment. 1 and 2 represent the first 10 and the last 11 features respectively in higgs. In the case of susy, 1 and 2 represents first 4 and last 4 features, respectively.'
8. `--interval_set` - 1 represent first 200k instances, 2 represent the next 200k instances, and so on. This is also for higgs and susy datset for packetLSTM-retraining expriment.

### Run Example

#### For Benchmarking and ablation studies:
```
python Code/run_model.py --nruns 5 --seed 2024 --dataset magic04 --syn_data_type variable_p --availprob 0.75 --exp_num 101
```
You can change exp_num to perform other experiments on magic04 dataset with p = 0.75. Similarly, you can change the dataset and availprob to run on other dataset combination. 

In the case of real dataset (imdb), `--syn_data_type variable_p --availprob 0.75` is not required.

#### For Sudden, Obsolete, and Reappearing Experiments
We execute the sudden, obsolete, and reappearing experiments for SUSY and HIGGS datasets with corresponding exp_num as 201, 202, and 203, respectively.
```
python Code/run_model.py  --nruns 5 --seed 2024 --dataset susy --syn_data_type sudden --exp_num 201
```
Change dataset, syn_data_type, and exp_num to run other experiments.

#### packetLSTM-Retraining at Each Interval
This is to run packetLSTM from scratch at each data interval as explained in the paper. Here too, experiments are performed on SUSY and HIGGS datasets. The parameters for each experiment are:
1. first interval:  --exp_num 301 --feature_set 1 --interval_set 1
2. second interval: --exp_num 302 --feature_set 2 --interval_set 2
3. third interval:  --exp_num 303 --feature_set 1 --interval_set 3
4. fourth interval: --exp_num 304 --feature_set 2 --interval_set 4
5. fifth interval:  --exp_num 305 --feature_set 1 --interval_set 5

Code to run:
```
python Code/run_model.py --nruns 5 --seed 2024 --dataset susy --syn_data_type reappearing --feature_set 1 --interval_set 1 --exp_num 301
```

#### Dropping Features to Resolve Space Complexity of packetLSTM
Here, we define the maximum limit on the number of LSTMs and the number of times a feature is seen to 100. \
Code to run:
```
python Code/run_scalability.py --nruns 5 --seed 2024 --dataset imdb --exp_num 101 --feature_limit 100 --min_feature_instances 100
```

#### Determining best aggregation operators between Sum and Mean
Code for Sum:
```
python Code/run_model.py --nruns 5 --seed 2024 --dataset a8a --syn_data_type alternating_variable_p --availprob 0 --exp_num 1001
```

For mean: Change --exp_num to 1002

### Reading the results

After Executing the code,
1. All the benhcmark and ablation studies results will be stored at `Code/Results/<data_name>/prob_<availprob>/` for synthetic datasets and `Code/Results/<data_name>/` for real dataset (imdb).
2. The Sudden, Obsolete, and Reappearing Experiments results will be stored at `Code/Results/<data_name>/p75/`

We provide two files to read results: `read_results.py` for benchmark and ablation studies and `read_results_challenging_scenario.py` for other experiments.

#### For reading benchmark and ablation studies result
Use `read_results.py` with following parameters:
1. `--dataset` - Name of the dataset
2. `--availprob` - Value of probabilty
3. `--exp_num` - value of the single experiment to read results of

Code to run:
```
python Code/read_results.py --dataset magic04 --availprob 0.75 --exp_num 101
```

#### For reading other experiments
Use `read_results_challenging_scenario.py` with following parameters:
1. `--dataset` - Name of the dataset
2. `--syn_data_type` - Class of Synthetic Data
3. `--nruns` - Number of model runs
4. `--exp_num` - value of the single experiment to read results of

Code to run:
```
python Code/read_results_challenging_scenario.py --dataset susy --syn_data_type sudden --nruns 5 --exp_num 201
```


## Running Baselines
Refer to the `Baselines` Folder.

Basline Models:
1. NB3, FAE, OLVF, OCDS, OVFM, DynFo, ORF3V, Aux-Net, Aux-Drop - Used code from https://github.com/Rohit102497/HaphazardInputsReview
2. OLIFL - Adapted code from https://github.com/youdianlong/OLIFL

### Control Parameters
For each baseline model the parameters defer. All the parameters for each model is provided in `Baselines/Config/config.py`. Here, unlike packetLSTM code, exp_num is not embedded in config, since ablation studies are not needed. To play around with any baseline model, just change the value of corresponding model parameters is `Baselines/Config/config.py` file.

### Running Code Parameters

To run any baseline, we need to pass below parameters with the file `Baselines/main.py`

1. `--seed` : Seed value  
2. `--exp_num` - Provide an exp_num value. Your experiment result will be saved corresponding to this exp_num value. While reading the results this value is needed.
3. `--dataset`: The name of the dataset  
4. `--syn_data_type`: The type to create suitable synthetic dataset  
5. `--availprob`: The probability of each feature being available to create synthetic data    
6. `--methodname`: The name of the method (model)  
7. `--ifimputation`: If some features needs to be imputed. Needed for Aux-Net
8. `--imputationtype`: The type of imputation technique to create base features. Needed for Aux-Net
9. `--nimputefeat`: The number of imputation features. Needed for Aux-Net  
10. `nruns`: The number of times a method should runs (For deterministic method, it would be one.)  

### Run Example

#### For Benchmarking:
For NB3, FAE, OLVF, OCDS, OVFM, DynFo, ORF3V, OLIFL, Aux-Drop
```
python Baselines/main.py --exp_num 101 --dataset magic04 --syn_data_type variable_p --availprob 0.75 --methodname ovfm
```

For Aux-Net
```
python Baselines/main.py --exp_num 101 --dataset magic04 --syn_data_type variable_p --availprob 0.75 --methodname auxnet --ifimputation True
```

In the case of real dataset (imdb), `--syn_data_type variable_p --availprob 0.75` is not required.

#### For Sudden, Obsolete, and Reappearing Experiments
We executed the sudden, obsolete, and reappearing experiments for SUSY and HIGGS datasets with OLVF, OVFM, OLIFL, and Aux-Drop baselines.
```
python Baselines/main.py --exp_num 201 --dataset susy --syn_data_type sudden --methodname olvf
```
Change dataset, syn_data_type, and methodname to run other experiments.


### Reading the results

After Executing the code,
1. All the benchmark results will be stored at `Baselines/Results/noassumption/<methodname>/<data_name>_prob_<availprob>/` for synthetic datasets and `Results/noassumption/<methodname>/<data_name>/` for real dataset (imdb).
2. The Sudden, Obsolete, and Reappearing Experiments results will be stored at `Baselines/Results/noassumption/<methodname>/<data_name>/`


We provide two files to read results: `read_results.py` for benchmark and ablation studies and `read_results_challenging_scenario.py` for other experiments.

#### For reading benchmark and ablation studies result
Use `read_results.py` with following parameters:
1. `--dataset` - Name of the dataset
2. `--availprob` - Value of probabilty
3. `--methodname` - The name of the method (model) 
3. `--exp_num` - value of the single experiment to read results of

Code to run:
```
python Baselines/read_results.py --dataset magic04 --availprob 0.75 --methodname ocds --exp_num 101
```

#### For reading other experiments
Use `read_results_challenging_scenario.py` with following parameters:
1. `--dataset` - Name of the dataset
2. `--syn_data_type` - Class of Synthetic Data
3. `--nruns` - Number of model runs
4. `--exp_num` - value of the single experiment to read results
5. `--methodname` - The name of the method (model) 

Code to run:
```
python Baselines/read_results_challenging_scenario.py --dataset susy --syn_data_type sudden --methodname auxdrop --exp_num 201
```

## Running single LSTM

### Run Example
Code to run:
```
python Code/run_single_lstm.py --dataset magic04 --nruns 5 --syn_data_type variable_p --exp_num_list 101 --availprob 0.5 --exp_type Imputation --impute_type forward_fill
```
The configuration is stored in `Code/Config/singlelstm_config.py`. \
For Gaussian Copula, go to file `Code/Utils/imputation.py` and uncomment the code corresponding to Gaussian Copula.

### For reading result
Use `Code/read_results_single_lstm.py` with following parameters:
1. `--dataset` - Name of the dataset
2. `--availprob` - Value of probabilty
4. `--exp_num` - value of the single experiment to read results of
5. `--exp_type` - Type of experiment run. Imputation.
6. `--impute_type` - The type of imputation to use. 

Code to run:
```
python Code/read_results_single_lstm.py --dataset magic04 --availprob 0.75 --exp_num 101 --exp_type Imputation --impute_type forward_fill
```

## Running Transformer

### Padding

#### Running Models
Config is avaialble at `Code/Config/online_transformer_config.py`. \
Code to run:
```
python Code/run_transformer.py --dataset magic04 --nruns 5 --availprob 0.5 --exp_num_list 101 --syn_data_type variable_p --exp_type InputPairs
```

Replace --exp_type to PaddedInputs for the other padding option.

#### Reading Results
Code:
```
python Code/read_result_transformer.py --dataset magic04 --availprob 0.5 --exp_num 101 --exp_type InputPairs
```

### Natural Language
#### Running Models 
Config is avaialble at `Code/Config/online_nlp_config.py`. \
Code to run:
```
python Code/run_nlp.py --dataset magic04 --nruns 5 --availprob 0.5 --exp_num_list 101 --syn_data_type variable_p --exp_type OnlyValues --model_type bert
```

Replace --exp_type to InputPairs for other inputs type option. \
Replace --model_type to distilbert for other model option.


#### Reading Results
Code:
```
python Code/read_result_nlp.py --dataset magic04 --availprob 0.5 --exp_num 101 --exp_type OnlyValues --model_type bert
```

### Set Transformer
#### Running Models 
Config is avaialble at `Code/Config/set_transformer_config.py`. \
Code to run:
```
python Code/run_set_transformer.py --dataset magic04 --nruns 5 --availprob 0.5 --exp_num_list 101 --syn_data_type variable_p --exp_type Encoder_Set
```

#### Reading Results
Code:
```
python Code/read_result_set_transformer.py --dataset magic04 --availprob 0.5 --exp_num 101 --exp_type Encoder_Set
```

### HapTransformer
#### Running Models 
Config is avaialble at `Code/Config/hapTransformer_config.py`. \
Code to run:
```
python Code/run_hapTransformer.py --dataset magic04 --nruns 5 --availprob 0.5 --exp_num_list 101 --syn_data_type variable_p --exp_type Encoder_Ind
```

#### Reading Results
Code:
```
python Code/read_result_hapTransformer.py --dataset magic04 --availprob 0.5 --exp_num 101 --exp_type Encoder_Ind
```