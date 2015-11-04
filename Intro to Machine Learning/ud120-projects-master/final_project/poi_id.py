#!/usr/bin/python3

import sys
import pickle
sys.path.append("../tools/")
import IPython
from tester import test_classifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', "other", "expenses", "bonus", "salary"]

### Load the dictionary containing the dataset
data_dict = pickle.load(open(r"data/final_project_dataset.pkl", "rb") )

### Task 2: Remove outliers
del data_dict['TOTAL']
del data_dict["THE TRAVEL AGENCY IN THE PARK"]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Engineer email_bool variable
bool_list = ["other", "expenses", "bonus"]
for person in data_dict:
    for feature in bool_list:
        if data_dict[person][feature] != 'NaN':
             data_dict[person][feature] = 1
        else:
            data_dict[person][feature] = 0

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, \
                        remove_any_zeroes=False, sort_keys = False)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans

preprocessor = SelectKBest(k=7)
scale = StandardScaler()
pca = PCA(n_components = 2)

sv_clf = LinearSVC(C=4, class_weight = {1:3.2, 0:1})
km_clf = KMeans(n_clusters=2)

clf = Pipeline([('pca', pca),('scaler', scale),('clf', sv_clf)])



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list)
