'''
IPython Exploratory Analysis of the data
'''
import pickle
import pandas as pd
import numpy as np
pd.set_option('display.width',180)

data = open(r'../data/final_project_dataset_modified_py3.p', 'rb')
d = pickle.load(data)

df = pd.DataFrame(d).transpose().replace({'NaN':np.nan})

#Check Dataframe Shape
df.shape