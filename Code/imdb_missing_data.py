# Find the % of unavaialable values in imdb datasets

from Data_Code.data_load import data_load_synthetic, data_load_real
import numpy as np
import sys

data_name = 'imdb'
X, Y, X_haphazard, mask = data_load_real(data_name)

n_value = mask.shape[0]*mask.shape[1]
print("Number of expected values: ", n_value)
print("Number of observed values:", np.sum(mask))
print("Missing percentage: ", np.round(100*(n_value - np.sum(mask))/n_value, 2)) 