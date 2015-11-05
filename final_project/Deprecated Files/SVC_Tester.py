'''
SVC looks like it has the most promise
Dedicated tester just for it and stratified shuffling
'''

'''
Design choices
Use KBest feature selection only so I can build into pipeline
'''

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(r'../tools')
import pandas_df_split
import IPython

from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit

#Load data:
data = pickle.load(open(r'data/final_project_dataset.pkl', 'rb'))

#Create dataframe from loaded data
#Features are columns
#Rows are names
df = pd.DataFrame(data).transpose()

#Replace all missing values with zero
#Planning on using classification the arbitrary numerical value shouldn't skew anything
df.replace({'NaN':np.NaN}, inplace = True)

#Remove erroneous total row
df.drop("TOTAL", inplace=True)

#Engineereed features
#All POI have email addresses, convert to 0 1 flag indicating if email exists
df["email_bool"] = df["email_address"].replace({0:None}).notnull().astype("int")

# Drop all email column
df.drop("email_address", axis = 1, inplace = True)

#Get Targets
y = df['poi'].copy()
df.drop('poi', axis = 1, inplace = True)

#Impute missing values with mean
#This is a big assumption so test it later
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

#Cross validator
cv = StratifiedShuffleSplit(y, 1000, random_state = 42)

# Start building Pipeline
kbest = SelectKBest()
scale = StandardScaler()
clf = SVC()

#Set Params
params = {'feature_select__k_features':[2,5,7,],
            'clf__C':[.1,1,50,100],
            'clf__kernel':['linear']
             }

p = Pipeline([('feature_select', kbest),('scaler', scale),('clf', clf)])
grid = GridSearchCV(p, param_grid = params, scoring = 'f1', verbose = 10, cv=cv)
grid.fit(y, df.values)


#Print Parameters
print(grid.best_estimator_.named_steps["feature_select"].kbest.pvalues_)
print(grid.best_params_)
pred = grid.predict(X_test)
score_metrics(pred)
