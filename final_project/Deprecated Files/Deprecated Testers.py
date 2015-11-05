'''
Moving old classifier code to here
'''


rf_clf = RandomForestClassifier(n_jobs = 4, random_state = 42)
dt_clf = DecisionTreeClassifier(random_state = 42)
gb_clf = GaussianNB()

rf_classifier = Pipeline([('feature_select', preprocessor), ('clf', rf_clf)])
dt_classifier = Pipeline([('feature_select', preprocessor), ('clf', dt_clf)])
gb_classifier = Pipeline([('feature_select', preprocessor),('clf', gb_clf)])

rf_params = {
    'feature_select__k':[1,2,3],
    'rf__n_estimators':[5,10,20,30]
    }

dt_params = {
    'feature_select__k':[1,2,3,4,5],
    'clf__min_samples_split':[2,3,4],
    'clf__criterion':['gini'],
    'clf__max_depth':[2,3,4]
    }
gb_params = {
    'feature_select__k':[1,2,3]
}