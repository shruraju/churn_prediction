import pandas as pd 

df = pd.read_csv(r"C:\Users\Thirumathi\Documents\Learnbay docs\Churn_prediction_project_aws\data\Telco-Customer-Churn.csv")

#preview first 5 rows
print(df.head())

print("dataset info")
print(df.info(), "\n")

#check for missing values
print("missing values in each column")
print(df.isnull().sum(), "\n")

print("description statistics")
print(df.describe())

#target variable distribution
if 'Churn' in df.columns:
    print("\n==== Target variable Distrbution (Churn) ====")
    print(df['Churn'].value_counts())
else:
    print("\nChurn column not found")

#check unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\n ***Unique values in categorical column***")
for col in categorical_cols:
    print(f"{col}:{df[col].unique()}")

numeric_cols=df.select_dtypes(include=['int64', 'float64']).columns
print("\n======== Skewness of numeric features ========")
print(df[numeric_cols].skew())