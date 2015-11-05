'''
Pandas tester for sklearn

First iteration didn't use a pipeline
'''
import pandas as pd
import pickle
import sys
sys.path.append(r'../tools')
import pandas_df_split
import IPython
import ipdb

# ipdb.set_trace()
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
df["email_bool"] = df["email_address"].replace({0:None}).notnull().astype("int")

#Get test train split
X_train, y_train, X_test, y_test = pandas_df_split.df_test_train_split(df)
# IPython.embed()
#Human feature selection first
features_list = ["deferred_income", "director_fees", \
        "restricted_stock", "other", "exercised_stock_options", "email_bool"]

class feature_select:
    '''
    Feature Selection Class for Enron Dataset
    '''

    def __init__(self, features_list, k_features):
        '''
       Initialize Enron Preprocessor
        Pass list of human selected features

        Returns the union of the human selected and SelectKBest features
        '''

        self.features_list = features_list
        self.k_bool = None
        self.k_features = k_features

    def fit(self, X, y):
        '''
        Fit the K means estimator for test data
        '''
        #Find the best features from the dataframe
        from sklearn.feature_selection import SelectKBest
        self.kbest = SelectKBest(k = self.k_features)

        # Find the K Best features
        self.k_bool = self.kbest.fit(X.values, y.values).get_support()

        return self

    def transform(self, X):
        # ipdb.set_trace()
        #Get user selected columns
        self.user_bool = X.columns.isin(features_list)

        self.column_mask = self.user_bool | self.k_bool
        print(X.columns[self.column_mask])

        return X.loc[:,self.column_mask]

# Features Selection and preprocess
preprocessor = feature_select(features_list, k_features = 2)
preprocessor.fit(X_train, y_train)
df_sub = preprocessor.transform(X_train)
# ipdb.set_trace()

# Pick model and fit model
# Gonna try random forest

from sklearn.ensemble import RandomForestClassifier

# Fit a classifier
clf = RandomForestClassifier(n_jobs = 4, random_state = 42)
clf.fit(df_sub.values, y_train.values)

#Predict values
pred = clf.predict(preprocessor.transform(X_test))

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
acc = accuracy_score(y_test, pred)

print("The accuracy is {0}".format(acc))

confusion = confusion_matrix(y_test, pred, labels = [True, False])