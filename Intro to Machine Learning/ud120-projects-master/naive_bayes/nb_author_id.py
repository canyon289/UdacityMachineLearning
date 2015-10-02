#!/usr/bin/python3

"""
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project

    use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Create Naive Bayes Model
model = GaussianNB()

# Train model on training data
model.fit(features_train, labels_train)

# Assembly predictions
pred = model.predict(features_test)
print(accuracy_score(labels_test, pred))
