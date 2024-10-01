import numpy as np
from queue import Queue

# filling with the last seen value
def forward_fill_imputation(X_haphazard, mask):
    print("Using Forward Fill Imputation")
    mask = mask.astype(int).astype(bool)
    forward_fill = np.zeros(X_haphazard.shape[1])
    for i in range(X_haphazard.shape[0]):
        forward_fill[mask[i,]] = X_haphazard[i, mask[i,]]
        X_haphazard[i, ] = forward_fill

    return X_haphazard

def knn_mean_imputation(X_haphazard, mask, k = 5):
    print("Using knn Mean Imputation")
    mask = mask.astype(int).astype(bool)
    data_storage = [Queue(maxsize=k) for i in range(X_haphazard.shape[1])]

    for i in range(X_haphazard.shape[0]):
        idx = np.where(mask[i, ] == 1)
        for j in idx[0]:
            if data_storage[j].full():
                data_storage[j].get()
            data_storage[j].put(X_haphazard[i, j])
        idx_not = np.where(mask[i, ] == 0)
        for j in idx_not[0]:
            if data_storage[j].full():
                X_haphazard[i, j] = np.mean(list(data_storage[j].queue))

    return X_haphazard

# from gcimpute.gaussian_copula import GaussianCopula
# def gaussian_copula_imputation(X_haphazard, mask):
# # https://github.com/udellgroup/gcimpute/blob/master/Examples/Main_Tutorial.ipynb
# # https://github.com/yuxuanzhao2295/Online-Missing-Value-Imputation-and-Change-Point-Detection-with-the-Gaussian-Copula/tree/main
#     print("Using Gaussian Copula Imputation")
#     batch_size = 5
#     window_size = 5
#     seed = 2024
#     const_stepsize = 0.5
#     mask = mask.astype(int).astype(bool)
#     a = np.empty(X_haphazard.shape)
#     a.fill(np.nan)
#     X_haphazard = np.where(mask, X_haphazard, a)
#     gc = GaussianCopula(training_mode='minibatch-online', 
# 						    const_stepsize=const_stepsize, 
# 						    batch_size=batch_size, 
# 						    random_state=seed, 
# 						    n_jobs=1,
# 						    window_size=window_size,
# 						    realtime_marginal=False,
#                             verbose = 0
# 						    )
#     X_haphazard = gc.fit_transform(X_haphazard)
#     return X_haphazard

def gaussian_copula_imputation(X_haphazard, mask):
    pass