import pickle
import pandas as pd

# load the model from disk
filename = 'rf_model.sav'
clf = pickle.load(open(filename, 'rb'))

# preprocessing for input
def preprocess(test_data):

	# Importing the dataset
	dataset = pd.read_csv('Breast Cancer Data.csv')
	X = dataset.iloc[:, 2:32].values

	# using same preprocessing while training
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X = sc.fit_transform(X)

	test_data = sc.transform(test_data)

	return test_data

def random_forest_predict(clf, test_data):
	t = time()
	test_data = preprocess(test_data)
	output = clf.predict(test_data)
	acc = clf.predict_proba(test_data)
	print("The running time: ",time()-t)

	return output, acc, time()-t;

	

	