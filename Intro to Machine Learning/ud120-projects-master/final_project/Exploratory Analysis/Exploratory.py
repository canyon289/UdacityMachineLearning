'''
IPython Exploratory Analysis of the data
'''
import pickle
import pandas as pd
import numpy as np

data = open(r'../data/final_project_dataset_modified.pkl', 'rb')
d = pickle.load(data)

df = pd.DataFrame(d).transpose().replace({'NaN', np.nan})

#Check Dataframe Shape
df.shape