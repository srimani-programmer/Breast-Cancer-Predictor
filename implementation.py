import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time

def random_forest_train():

	# Importing the dataset
	dataset = pd.read_csv('Breast Cancer Data.csv')
	X = dataset.iloc[:, 2:32].values
	y = dataset.iloc[:, 1].values

	# Encoding categorical data
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_X_1 = LabelEncoder()
	y = labelencoder_X_1.fit_transform(y)

	# Splitting the dataset into the Training set and Test set
	global X_test, y_test
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	global sc
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	return clf

def randorm_forest_test(clf):
	t = time()
	output = clf.predict(X_test)
	acc = accuracy_score(y_test, output) 
	print("The accuracy of testing data: ",acc)
	print("The running time: ",time()-t)

def random_forest_predict(clf, inp):
	t = time()
	inp = sc.transform(inp)
	output = clf.predict(inp)
	acc = clf.predict_proba(inp)
	print("The running time: ",time()-t)

	return output, acc, time()-t;