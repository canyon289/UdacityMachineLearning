'''
Pandas tester for sklearn
'''
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split


#Load data
data = pickle.load(open(r'data/final_project_dataset.pkl', 'rb'))

#Create dataframe from loaded data
#Features are columns
#Rows are names
df = pd.DataFrame(data).transpose()

#Replace all missing values with zero
#Planning on using classification the arbitrary numerical value shouldn't skew anything
df.replace({'NaN':0}, inplace = True)

#Remove erroneous total row
df.drop("TOTAL", inplace=True)

#Engineereed features
#All POI have email addresses, convert to 0 1 flag indicating if email exists
df["email_bool"] = df["email_address"].notnull().astype("int")

#Get test train split
train_indices, test_indices = train_test_split(df.index.values, test_size = .3)

#Set target and drop from dataframe
target = df["poi"].copy()
df.drop(["poi", "email_address"])", axis = 1, inplace = True)

#Human feature selection first
features_list = ["deferred_income", "director_fees", \
        "restricted_stock", "other", "exercised_stock_options", "email_bool"]

class feature_select:
    '''
    Feature Selection Class for Enron Dataset
    '''
    
    def __init__(self):
        self.k_bool = None
        
    def fit(self, df, train_indices, features_list, k_features):
        '''
        Fit the K means estimator for test data
        '''
        #Find the best features from the dataframe
        from sklearn.feature_selection import SelectKBest
        self.kbest = SelectKBest(k = k_features)
        self.k_bool = self.kbest.fit(df.loc[train_indices,:].values, target.loc[train_indices,:].values).get_support()
        
        return self
    
    def transform(self, df):
        
        #Get user selected columns
        self.user_bool = df.columns.isin(feature_list)
        
        transform_bool = self.user_bool | self.k_bool
        print(df.columns[transform_bool])
        
        return df[transform_bool]

preprocessor = feature_select()
preprocessor.fit(df, train_indices, features_list, k_features)
df = preprocessor.transform(df)