# test nn vs tensorflow
import matplotlib.pyplot as plt
from nn import NN 
import numpy as np
import pandas as pd 
# load data 
data_path = "./data/breast_cancer.csv"
data = pd.read_csv(data_path)
data = data.values
print(data.shape)