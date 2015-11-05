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