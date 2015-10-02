'''
Udacity Quiz for Classifer
'''

def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!

    # Import sklearn algorithm
    from sklearn.naive_bayes import GaussianNB

    # Create classifer object
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    return clf
