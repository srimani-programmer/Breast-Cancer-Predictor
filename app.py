from flask import Flask, render_template, request
from implementation import randorm_forest_test, random_forest_train, random_forest_predict
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_forest import accuracy
from sklearn.metrics import accuracy_score
from time import time
import logging
import os

app = Flask(__name__)
app.url_map.strict_slashes = False

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir="applicationLogs"
general_logs = "logs"
general_log_path_dir=os.path.join(log_dir,general_logs)

os.makedirs(general_log_path_dir, exist_ok=True)
general_logs_name = "general_logs.log"
general_log_path = os.path.join(general_log_path_dir,general_logs_name)
print(general_log_path)
logging.basicConfig(filename = general_log_path, level=logging.INFO, format=logging_str)
logging.info("Application started")

@app.route('/')
def index():
	logging.info("Index page loaded")
	return render_template('home.html')

@app.route('/redirect', methods=['POST']) 
def redirectionform():
	logging.info("Redirection form loaded")
	return render_template('form.html')

@app.route('/backtohome', methods=['POST']) 
def a():
	logging.info("Redirection to home page")
	return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def login_user():
	logging.info("Prediction form loaded")
	data_points = list()
	logging.info("Data points list created")
	data = []
	string = 'value'
	for i in range(1,31):
		data.append(float(request.form['value'+str(i)]))

	for i in range(30):
		data_points.append(data[i])
		
	print(data_points)
	logging.info("Data points list appended")
	data_np = np.asarray(data, dtype = float)
	logging.info("Data points list converted to numpy array")
	data_np = data_np.reshape(1,-1)
	logging.info("Data points list reshaped")
	out, acc, t = random_forest_predict(clf, data_np)
	logging.info("Prediction done")

	if(out==1):
		output = 'Malignant'
	else:
		output = 'Benign'

	acc_x = acc[0][0]
	acc_y = acc[0][1]
	if(acc_x>acc_y):
		acc1 = acc_x
	else:
		acc1=acc_y
	return render_template('result.html', output=output, accuracy=accuracy, time=t)

	

if __name__=='__main__':
	global clf 
	clf = random_forest_train()
	randorm_forest_test(clf)
	#print("Done")
	app.run(debug=True)
