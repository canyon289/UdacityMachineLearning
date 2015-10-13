#!/usr/bin/python3


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error)
    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(list(zip(predictions, ages, net_worths)), columns = ["Pred", "Age", "Net Worth"])
    df["diff"] = np.absolute(df["Pred"] - df["Net Worth"])
    df.sort("diff", inplace = True).reset_index()

    df_sub = df.iloc[81:,:]
    cleaned_data =  list(zip(df_sub["age"], df_sub["Net Worth"], df_sub["diff"]))

    ### your code goes here


    return cleaned_data

