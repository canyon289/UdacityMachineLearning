'''
IPython Exploratory Analysis of the data
'''
import pickle
import pandas as pd
import numpy as np
import IPython

data = open(r'../data/final_project_dataset_modified_py3.p', 'rb')
d = pickle.load(data)

df = pd.DataFrame(d).replace({'NaN':np.nan}).transpose()

#Check Dataframe Shape
df.shape
IPython.embed()