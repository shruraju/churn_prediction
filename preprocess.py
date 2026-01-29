import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Thirumathi\Documents\Learnbay docs\Churn_prediction_project_aws\data\Telco-Customer-Churn.csv")

print("original dataset shape", df.shape)

#Handling missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Drop rows with missing values
df.dropna(inplace=True)
print("after dropping missing values", df.shape) 

duplicates = df.duplicated().sum()
print("number of duplicate rows:", duplicates)

if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

#=============================================================
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f'{col} vs Churn')
    plt.show()

df['Churn']=df['Churn'].map({'Yes':1, 'No':0})

#encode categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#split into features and target
X=df.drop('Churn',axis=1)
y=df['Churn']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("train_shape:", X_train.shape)
print("test_shape:", y_test.shape)

