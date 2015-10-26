'''
Pandas tester for sklearn
'''
import pandas as pd
import pickle
import sys
sys.path.append(r'../tools')
import pandas_df_split
import IPython
import ipdb
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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

# Now I need to build a pipeline and see how that works
print("Testing pipeline methods to see how it goes")
# Reset feature selection and model
preprocessor = feature_select(features_list, k_features = 2)
clf = RandomForestClassifier(n_jobs = 4, random_state = 42)

preprocessor = feature_select(features_list, k_features = 2)
rf_classifier = Pipeline([('feature_select',preprocessor), ('rf', clf)])

rf_classifier.fit(X_train, y_train)
pred_pipeline = rf_classifier.predict(X_test)

acc = accuracy_score(y_test, pred_pipeline)
print("Recall:{0}".format(recall_score(y_test, pred_pipeline)))
print("Precision:{0}".format(precision_score(y_test, pred_pipeline)))
print("The accuracy is {0}".format(acc))
'''
Pipelines aren't so bad

So what I need to do to make a model
Get data
Do some exploration
Split the data into test and train pairs
Set up a preprocessor with Fit and Transform methods
Setup a pipeline with tests
Run the pipeline with transform and predict

#Things left to do
Clean up main file
Train another type of model
Run a Grid CV Search
Figure out what works best

Open Questions
Can grid search only fit best parameters on fit or also on predict?
'''
# IPython.embed()
