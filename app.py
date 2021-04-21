import os
from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
#from implementation import randorm_forest_test, random_forest_train, random_forest_predict
import implementation
#from sklearn.preprocessing import StandardScaler
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from random_forest import accuracy
#from sklearn.metrics import accuracy_score
from time import time

# Importing the database file from the SQLITE database ---------------------------------------------- o
project_dir = os.path.dirname(os.path.abspath(__file__))										     #|
database_file = "sqlite:///{}".format(os.path.join(project_dir, "Breast-Cancer-Predictor-Users.db")) #|
# --------------------------------------------------------------------------------------------------- o

app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['SQLALCHEMY_DATABASE_URI'] = database_file

# Initializing the reference variable for the database connect -o
db = SQLAlchemy(app)										   #|
# --------------------------------------------------------------o

# Initializing the User data Model ----------------------o
class User(db.Model):									#|
    name = db.Column(db.String(30), nullable=False)     #|
    email = db.Column(db.String(40), primary_key=True)  #|
# -------------------------------------------------------o



@app.route('/')
def index():
	return render_template('home.html')

# BELOW, USER DATA HAS BEEN ADDED 
@app.route('/insert', methods=['POST','GET'])
def get_to_know():
	user = User(name=request.form['name'], email=request.form['email'])
	if user.name == '' and user.email == '':
		return redirect('/')
	else:
		db.session.add(user)
		db.session.commit()
		return redirect('/')


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
	out, acc, t = random_forest_predict(clf, data_np)

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
	db.create_all()
	global clf 
	clf = random_forest_train()
	randorm_forest_test(clf)
	#print("Done")
	app.run(debug=True)

