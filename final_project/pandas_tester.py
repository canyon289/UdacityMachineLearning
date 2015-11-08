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
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


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
df.drop(["TOTAL", "THE TRAVEL AGENCY IN THE PARK"], inplace=True)

#Engineereed features
#All POI have email addresses, convert to 0 1 flag indicating if email exists
bool_list = ["other", "expenses", "bonus"]
for col in bool_list:
    name = col + "_bool"
    df[name] = df[col].notnull().astype("int")

# Drop all email column
df.drop("email_address", axis = 1, inplace = True)

#Different value imputers
#df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
df.replace({np.NaN:0}, inplace = True)

#Features List
features = df.columns

# Get test train split
X_train, y_train, X_test, y_test = pandas_df_split.df_test_train_split(df)
print(features)
IPython.embed()


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


# Reset feature selection and model
pca = PCA()
feature_select = SelectKBest()
scale = StandardScaler()

#Be sure to check importer to get rid of replace if using Gaussian
sv_clf = LinearSVC()
km_clf = KMeans(max_iter = 200)
lr_clf = LogisticRegression(class_weight = {1:10, 0:1})


sv_classifier = Pipeline([('scaler', scale),('feature_select', feature_select), ('pca', pca), ('clf', sv_clf)])
km_classifier = Pipeline([('scaler', scale),('feature_select', feature_select), ('pca', pca),('clf', km_clf)])
lr_classifier = Pipeline([('scaler', scale),('feature_select', feature_select), ('pca', pca),('clf', lr_clf)])

sv_params = {'feature_select__k':list(range(1,len(df.columns))),
            'pca__n_components': [8,7,6,5,4,3,2],
            'clf__C':[.1,.5,1,50],
            'clf__class_weight' : ['auto', {1:3.25, 0:1}]
             }
             
km_params = {'feature_select__k':list(range(1,len(df.columns))),
            'pca__n_components': [8,7,6,5,4,3,2],
            'clf__n_clusters':[2]
             }
             
lr_params = {'feature_select__k':list(range(1,len(df.columns))),
            'pca__n_components': [8,7,6,5,4,3,2],
             }

grid = GridSearchCV(lr_classifier, param_grid = lr_params, scoring = 'f1', cv = 10, error_score = 0)
grid.fit(X_train, y_train)

print(grid.best_params_)
pred = grid.predict(X_test)
score_metrics(pred)

#IPython.embed()
