
<img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/Breast%20Cancer%20Prediction%20banner.png">

# Predicts whether the type of breast cancer is Malignant or Benign

![Issues](https://img.shields.io/github/issues/srimani-programmer/Breast-Cancer-Predictor)
![Pull Requests](https://img.shields.io/github/issues-pr/srimani-programmer/Breast-Cancer-Predictor)
![Forks](https://img.shields.io/github/forks/srimani-programmer/Breast-Cancer-Predictor)
![Stars](https://img.shields.io/github/stars/srimani-programmer/Breast-Cancer-Predictor)
[![License](https://img.shields.io/github/license/srimani-programmer/Breast-Cancer-Predictor)](https://github.com/srimani-programmer/Breast-Cancer-Predictor/blob/master/LICENSE)

> ## :round_pushpin: Please follow the [Code of Conduct](https://github.com/srimani-programmer/Breast-Cancer-Predictor/blob/master/CODE_OF_CONDUCT.md) for contributing in this repository
# :dart: Aim of the Project
####  -To predict if a breast cancer is Malignant or Benign using Image Dataset as well as Numerical Data
####  -Apply ML and DL Models to predict the severity of the Breast-Cancer
####  -Create a Wonderful UI for this project using Front End Languages and Frameworks (Like Bootstrap)
####  -Create the Backend using Flask Framework.
####  -Deploy on Cloud and make this wonderful project available to public


## :clipboard: Table of contents
* [About Project](#about-project)
* [Languages or Frameworks Used](#languages-or-frameworks-used)
* [Setup](#project-setup)
* [How To Contribute ?](https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/Contribution-Guide.md)
* [Application UI](#Application-ui)

## :large_blue_circle: About Project:

Breast cancer is the **most common type of cancer in women**. When cancers are found early, they can often be cured. 
There are some devices that detect the breast cancer but many times they lead to false positives, which results 
is patients undergoing painful, expensive surgeries that were not even necessary. These type of cancers are called 
**benign** which do not require surgeries and we can reduce these unnecessary surgeries by using Machine Learning. 
I have taken the dataset of the previous breast cancer patients and train the model to predict whether the cancer is **benign** or **malignant**. These predictions will help doctors to do surgeries only when the cancer is malignant, thus reducing the unnecessary surgeries for woman. 



For building the project I have used Wisconsin Breast cancer data which has 569 rows of which 357 are benign and 212 are malignant. 
The data is prepossessed and scaled.

The data set that I used is shown here.

<div align="center">
<h4>Data_Set</h4>
</div>

![Data_Set](https://github.com/tanya162/Breast-Cancer-Predictor/blob/New_Pipeline/static/images/img_dataset.png)
 
 ***** 

 To find the correalaton map was the second step.

<div align="center">
<h4>CORRELATION MAP</h4>
</div>

![CORRELATION MAP](https://github.com/tanya162/Breast-Cancer-Predictor/blob/New_Pipeline/static/images/breast_cancer%2Cdata_analysis_1.png)
 
 ***** 

I also figured out the most important features:


<div align="center">
<h4>Most Important Features</h4>
</div>

![Important Features](https://github.com/tanya162/Breast-Cancer-Predictor/blob/New_Pipeline/static/images/data_analysis_2.png)

***** 
The conacve points that I found are as follows.

<div align="center">
<h4>Concave Points</h4>
</div>

![Concave Points](https://github.com/tanya162/Breast-Cancer-Predictor/blob/New_Pipeline/static/images/concave_points.png)

*****

These steps complete our data visualization part and we are ready to use the data for machine learning algorithms.
***** 
*****


 I have trained the model with Random forest Classifier algorithm that gives an accuracy of 95.0%. To provide the easy to use interface to doctors I have developed a website that will take the data and display the output with accuracy and time taken to predict the same.

<img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/Description-image.png">


## :large_blue_circle: Languages & Frameworks Used 

  * Python: language
  * NumPy: library for numerical calculations
  * Pandas: library for data manipulation and analysis
  * SkLearn: library which features various classification, regression and clustering algorithms
  * Flask: microframework for building web applications using Python.
  
  <img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/python-icon.png" width=100 height=100> <img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/numpy-icon.png" width=100 height=100> <img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/pandas-icon.png" width=100 height=100> <img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/sklearn-icon.png" width=100 height=100>  <img src="https://github.com/M-PRERNA/Breast-Cancer-Predictor/blob/master/static/images/flask-icon.png" width=100 height=100>
  
## :large_blue_circle: Project Setup
  To setup this project on your systems
  * First Clone the repository.
  * Create the virtual environment for the project. 
  
  :heavy_check_mark: **FOR ANACONDA USERS**
  ```sh
  $ conda create -n myenv python=3.6
  ```
  :heavy_check_mark: **FOR VENV USERS**
  - Create a new directory (ex: project)
  - Move into *project* directory
  - type the following:
  ```sh
  python -m venv my_virtualenv
  ```
  * After activating the virtual environment, install the required packages using requirements.txt inside the environemnt using pip.
  ```sh
  $ pip install -r requirements.txt
  ```
  * run the app.py as `python app.py`
  * Web Application will be hosted at  `127.0.0.1:5000`
  * Enter the URL in the browser Application will be hosted.
  * Enter the details of the tumor to detect the type of the cancer with more than 95% accuracy.

<div align="center">
<h1> :large_blue_circle: Application UI</h1>
</div>

<div align="center">
<p> :house: Home Page<p>
</div>

<img src="static/images/homepage1.png" width="1200" height="600">

<div align="center">
<p> :clipboard: Tumor Data form</p>
</div>

<div align=""center>
<img src="static/images/HomePage2.png" width="1200" height="600">
</div>

<div align="center">
<p> :clipboard: Tumor Data form contnd.</p>
</div>

<img src="static/images/homepage3.png" width="1200" height="600">

<div align="center">
<p> :mag: Prediction Output</p>
</div>

<img src="static/images/predict.png" width="1200" height="600">

## :star2: Awesome contributors :star2:
<a href="https://github.com/M-PRERNA/Breast-Cancer-Predictor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=M-PRERNA/Breast-Cancer-Predictor" />
</a>
 
*Made with [contributors-img](https://contributors-img.web.app).*
