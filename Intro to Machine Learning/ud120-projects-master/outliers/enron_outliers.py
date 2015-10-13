#!/usr/bin/python3

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
import pandas as pd
df = pd.DataFrame(data, columns = features)
df.plot(kind = 'scatter', x = "salary", y = 'bonus')

