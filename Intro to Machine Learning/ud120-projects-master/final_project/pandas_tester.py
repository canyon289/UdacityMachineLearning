'''
Pandas tester for sklearn
'''
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(r'../tools')
import pandas_df_split
import IPython
import ipdb
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            recall_score, precision_score, f1_score,\
                            make_scorer, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True)

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

# IPython.embed()
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)


# Get test train split
X_train, y_train, X_test, y_test = pandas_df_split.df_test_train_split(df)
# IPython.embed()
#Human feature selection first
features_list = ["other", "exercised_stock_options"]

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

        self.column_mask = self.k_bool | self.user_bool
        self.cols = X.columns[self.column_mask]

        return X.loc[:,self.column_mask]

    def get_params(self, deep=True):
        '''
        Need a get_params for GridSearchCV function
        '''
        return {"features_list":self.features_list, "k_features":self.k_features}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self

def score_metrics(predictions):
    '''
    Takes the list of predictions and outputs the various scores
    '''
    acc = accuracy_score(y_test, predictions)
    print("Recall:{0}".format(recall_score(y_test, predictions)))
    print("Precision:{0}".format(precision_score(y_test, predictions)))
    print("The accuracy is {0}".format(acc))
    print("Classification Report Matrix")

    print(classification_report(y_test, predictions, labels = [True, False]))
    return

# Now I need to build a pipeline and see how that works
print("Testing pipeline methods to see how it goes")
# Reset feature selection and model
preprocessor = feature_select(features_list, k_features = 2)
rf_clf = RandomForestClassifier(n_jobs = 4, random_state = 42)
dt_clf = DecisionTreeClassifier(random_state = 42)

#Be sure to check importer to get rid of replace if using Gaussian
gb_clf = GaussianNB()
sv_clf = SVC()
scale = StandardScaler()

#Create pipelines
preprocessor = feature_select(features_list, k_features = 2)
imputer = Imputer()
rf_classifier = Pipeline([('feature_select', preprocessor), ('clf', rf_clf)])
dt_classifier = Pipeline([('feature_select', preprocessor), ('clf', dt_clf)])
gb_classifier = Pipeline([('feature_select', preprocessor),('clf', gb_clf)])
sv_classifier = Pipeline([('feature_select', preprocessor),('scaler', scale),('clf', sv_clf)])
'''
for pipe in [rf_classifier, dt_classifier]:

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    score_metrics(predictions)
'''
# Make Scorer
f1 = make_scorer(f1_score)
#Random Forest looks most promising
rf_params = {
    'feature_select__k_features':[1,2,3],
    'rf__n_estimators':[5,10,20,30]
    }

dt_params = {
    'feature_select__k_features':[1,2,3,4,5],
    'clf__min_samples_split':[2,3,4],
    'clf__criterion':['gini'],
    'clf__max_depth':[2,3,4]
    }
gb_params = {
    'feature_select__k_features':[1,2,3]
}

sv_params = {'feature_select__k_features':[2,5,7,],
            'clf__C':[.1,1,50,100],
            'clf__kernel':['linear']
             }

grid = GridSearchCV(sv_classifier, param_grid = sv_params, scoring = 'f1', verbose = 10)
grid.fit(X_train, y_train)
'''
Pipelines aren't so bad

So what I need to do to make a model
Get data
Do some exploration
Split the data into test and train pairs
Set up a preprocessor with Fit and Transform methods
Setup a pipeline with tests
Run the pipeline with transform and predict
Clean up main file
Train another type of model

#Things left to do
Run a Grid CV Search
Figure out what works best

Open Questions
Can grid search only fit best parameters on fit or also on predict?
'''
feature_cols = grid.best_estimator_.named_steps["feature_select"].cols
# feature_importance = grid.best_estimator_.named_steps["gb"].feature_importances_
print(feature_cols)
print("Select K Best feature Importance")
print(grid.best_estimator_.named_steps["feature_select"].kbest.pvalues_)
# print("Decision Tree Feature Importance")
#for col,imp in zip(feature_cols, feature_importance):
#    print("{0}:{1}".format(col,imp))
print(grid.best_params_)
pred = grid.predict(X_test)
score_metrics(pred)
# IPython.embed()
