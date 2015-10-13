#!/usr/bin/python3


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import IPython

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]
data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size = .3, random_state = 42)
# IPython.embed()

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
# print(clf.score(features_test, labels_test))
print(recall_score(labels_test, clf.predict(features_test), labels=[1,0]))
print(precision_score(labels_test, clf.predict(features_test), labels=[1,0]))

