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


@app.route('/')
def index():
    values = ["Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean", "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean", "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE", "Compactness SE",
              "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE", "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst", "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Wymmetry Worst", "Fractal Dimension Worst"]
    return render_template('home.html', values=values)


@ app.route('/predict', methods=['POST'])
def login_user():

    data_points = list()
    data = []
    string = 'value'
    for i in range(1, 31):
        data.append(float(request.form['value'+str(i)]))

    for i in range(30):
        data_points.append(data[i])

    print(data_points)

    data_np = np.asarray(data, dtype=float)
    data_np = data_np.reshape(1, -1)
    out, acc, t = random_forest_predict(clf, data_np)

    if(out == 1):
        output = 'Malignant'
    else:
        output = 'Benign'

    acc_x = acc[0][0]
    acc_y = acc[0][1]
    if(acc_x > acc_y):
        acc1 = acc_x
    else:
        acc1 = acc_y
    return render_template('result.html', output=output, accuracy=accuracy, time=t)


if __name__ == '__main__':
    global clf
    clf = random_forest_train()
    randorm_forest_test(clf)
    # print("Done")
    app.run(debug=True)
