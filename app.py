from flask import Flask, render_template, request
from implementation import randorm_forest_test, random_forest_train, random_forest_predict
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_forest import accuracy
from sklearn.metrics import accuracy_score
from time import time


app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cancer-data-collection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Patient(db.Model):
    __tablename__ = "patient_data"
    id = db.Column(db.Integer, primary_key=True)
    
    radius_mean = db.Column(db.Float, nullable=False)
    texture_mean = db.Column(db.Float, nullable=False)
    peri_mean = db.Column(db.Float, nullable=False)
    area_mean = db.Column(db.Float, nullable=False)
    smooth_mean = db.Column(db.Float, nullable=False)
    compact_mean = db.Column(db.Float, nullable=False)
    concavity_mean = db.Column(db.Float, nullable=False)
    concave_p_mean = db.Column(db.Float, nullable=False)
    symmetry_mean = db.Column(db.Float, nullable=False)
    frac_dim_mean = db.Column(db.Float, nullable=False)
    
    radius_se = db.Column(db.Float, nullable=False)
    texture_se = db.Column(db.Float, nullable=False)
    peri_se = db.Column(db.Float, nullable=False)
    area_se = db.Column(db.Float, nullable=False)
    smooth_se = db.Column(db.Float, nullable=False)
    compact_se = db.Column(db.Float, nullable=False)
    concavity_se = db.Column(db.Float, nullable=False)
    concave_p_se = db.Column(db.Float, nullable=False)
    symmetry_se = db.Column(db.Float, nullable=False)
    frac_dim_se = db.Column(db.Float, nullable=False)
    
    radius_wr = db.Column(db.Float, nullable=False)
    texture_wr = db.Column(db.Float, nullable=False)
    peri_wr = db.Column(db.Float, nullable=False)
    area_wr = db.Column(db.Float, nullable=False)
    smooth_wr = db.Column(db.Float, nullable=False)
    compact_wr = db.Column(db.Float, nullable=False)
    concavity_wr = db.Column(db.Float, nullable=False)
    concave_p_wr = db.Column(db.Float, nullable=False)
    symmetry_wr = db.Column(db.Float, nullable=False)
    frac_dim_wr = db.Column(db.Float, nullable=False)

db.create_all()

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
	
#################################Adding the Database#####################################
	new_patient = Patient(

	    radius_mean = data[0],
	    texture_mean = data[1],
	    peri_mean = data[2],
	    area_mean = data[3],
	    smooth_mean = data[4],
	    compact_mean = data[5],
	    concavity_mean = data[6],
	    concave_p_mean = data[7],
	    symmetry_mean = data[8],
	    frac_dim_mean = data[9],

	    radius_se = data[10],    
	    texture_se = data[11],
	    peri_se = data[12],
	    area_se = data[13],
	    smooth_se = data[14],
	    compact_se = data[15],
	    concavity_se = data[16],
	    concave_p_se = data[17],
	    symmetry_se = data[18],
	    frac_dim_se = data[19],

	    radius_wr = data[20],    
	    texture_wr = data[21],
	    peri_wr = data[22],
	    area_wr = data[23],
	    smooth_wr = data[24],
	    compact_wr = data[25],
	    concavity_wr = data[26],
	    concave_p_wr = data[27],
	    symmetry_wr = data[28],
	    frac_dim_wr = data[29],

	)
    
	db.session.add(new_patient)
	db.session.commit()
#########################################################################################################

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
	global clf 
	clf = random_forest_train()
	randorm_forest_test(clf)
	#print("Done")
	app.run(debug=True)
