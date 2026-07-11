import pandas as pd
import numpy as np
import pickle
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("Dataset/Loan_prediction.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()
data['Gender'] = data['Gender'].map({'Female':1,'Male':0})

data['Property_Area'] = data['Property_Area'].map({
    'Urban':2,
    'Semiurban':1,
    'Rural':0
})

data['Married'] = data['Married'].map({
    'Yes':1,
    'No':0
})

data['Education'] = data['Education'].map({
    'Graduate':1,
    'Not Graduate':0
})

data['Loan_Status'] = data['Loan_Status'].map({
    'Y':1,
    'N':0
})
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

data['Married'] = data['Married'].fillna(data['Married'].mode()[0])

data['Dependents'] = data['Dependents'].str.replace('+','')

data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])

data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].map({
    'Yes': 1,
    'No': 0
})

data['Self_Employed'] = data['Self_Employed'].astype('int64')
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(
    data['Loan_Amount_Term'].mode()[0]
)

data['Credit_History'] = data['Credit_History'].fillna(
    data['Credit_History'].mode()[0]
)

data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])
data['Gender']=data['Gender'].astype('int64')

data['Married']=data['Married'].astype('int64')

data['Dependents']=data['Dependents'].astype('int64')







data['Gender']=data['Gender'].astype('int64')
data.info()
X = data.drop(['Loan_ID','Loan_Status'], axis=1)
y = data['Loan_Status']

smote = SMOTE(random_state=42)
x_bal, y_bal = smote.fit_resample(X, y)

print(y.value_counts())
print(y_bal.value_counts())

names = x_bal.columns
sc = StandardScaler()

x_bal = sc.fit_transform(x_bal)

x_bal = pd.DataFrame(x_bal, columns=names)
X_train, X_test, y_train, y_test = train_test_split(
    x_bal,
    y_bal,
    test_size=0.2,
    random_state=42
)

model = GradientBoostingClassifier()

model.fit(X_train, y_train)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(sc, f)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

print("Model and scaler saved successfully!")

