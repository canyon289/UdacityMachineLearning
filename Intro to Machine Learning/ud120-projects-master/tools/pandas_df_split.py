'''
Utility function that splits pandas dataframe into testing
and training sets
'''
from sklearn.cross_validation import train_test_split
import IPython
import ipdb

def df_test_train_split(df, target_col = "poi"):
    '''
    returns: X_train, y_train, X_test, y_test

    Splits Pandas dataframe into test and train set
    Splits based on rows

    '''
    train_indices, test_indices = train_test_split(df.index.values,
                                test_size = .3, random_state = 42)
    # Get target col and drop
    y = df[target_col].copy()
    df.drop(target_col, axis = 1, inplace = True)

    # ipdb.set_trace()


    # Make all four splits
    X_train =  df.loc[train_indices,:]
    y_train = y.loc[train_indices]
    X_test = df.loc[test_indices,:]
    y_test = y.loc[test_indices]

    return X_train, y_train, X_test, y_test