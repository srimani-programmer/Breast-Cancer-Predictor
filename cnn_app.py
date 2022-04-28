from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_forest import accuracy
from sklearn.metrics import accuracy_score
from time import time


app = Flask(__name__)
app.url_map.strict_slashes = False

# Model saved with Keras model.save()
MODEL_PATH = 'cnn_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/redirect', methods=['POST']) 
def redirectionform():
	return render_template('form.html')

@app.route('/backtohome', methods=['POST']) 
def a():
	return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def login_user():

	data_points = list()
	data = []
	string = 'value'
	for i in range(1,31):
		data.append(float(request.form['value'+str(i)]))

	for i in range(30):
		data_points.append(data[i])
		
	print(data_points)

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
    
	out = (model.predict(data_np) > 0.5).astype("int32")

	if(out==True):
		output = 'Malignant'
	else:
		output = 'Benign'

	return render_template('result.html', output=output,accuracy=0.969)

	

if __name__=='__main__':
	app.run(debug=True)
