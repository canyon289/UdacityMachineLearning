'''
Pandas tester for sklearn
'''

#Load data
data = pickle.load(open(r'/data/final_project_dataset_py3.p', 'rb'))

#Create dataframe from loaded data
#Features are columns
#Rows are names

df = pd.DataFrame(data).transpose()

#Replace all missing values with zero
#Planning on using classification the arbitrary numerical value shouldn't skew anything
df.replace({'NaN':0}, inplace = True)

#Remove erroneous total row
df.drop("TOTAL", inplace=True)

#Engineereed features
#All POI have email addresses, convert to 0 1 flag indicating if email exists
df["email_bool"] = df["email_address"].notnull().astype("int")


#Human feature selection first
selected_cols = ["deferred_income", "director_fees", \
        "restricted_stock", "other", "exercised_stock_options", "email_bool"]

selected_col_bool = df.columns.isin(select_cols)


# Select a K best features
# Possible issues might include the NAN values
from sklearn.feature_selection import SelectKBest

k_best_bool =
