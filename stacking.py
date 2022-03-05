
# Part 1 - Data Preprocessing

# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
dataset = pd.read_csv('Breast Cancer Data.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.metrics import accuracy_score
from time import time

from xgboost import XGBClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
lf = [('xgb',XGBClassifier(n_estimators=400,random_state=1,eval_metric='mlogloss')),('rfc',RandomForestClassifier()),('knn',  KNeighborsClassifier()
)]
clf = StackingClassifier( estimators = lf,final_estimator = DecisionTreeClassifier())
t = time()
clf.fit(X_train, y_train)
output = clf.predict(X_test)
accuracy = accuracy_score(y_test, output) 
print("The accuracy of testing data: ",accuracy)
print("The running time: ",time()-t)