# Breast Cancer Prediction
> Predicts whether the type of breast cancer is Malignant or Benign

## Table of contents
* [About Project](#about-project)
* [Languages or Frameworks Used](#languages-or-frameworks-used)
* [Setup](#setup)

## About Project:

Breast cancer is the most common type of cancer in women. When cancers are found early, they can often be cured. 
There are some devices that detect the breast cancer but many times they lead to false positives, which results 
is patients undergoing painful, expensive surgeries that were not even necessary. These type of cancers are called 
**benign** which do not require surgeries and we can reduce these unnecessary surgeries by using Machine Learning. 
I have taken the dataset of the previous breast cancer patients and train the model to predict whether the cancer is **benign** or **malignant**. These predictions will help doctors to do surgeries only when the cancer is malignant, thus reducing the unnecessary surgeries for woman. 

For building the project I have used Wisconsin Breast cancer data which has 569 rows of which 357 are benign and 212 are malignant. 
The data is prepossessed and scaled. I have trained with Random forest Classifier gives best accuracy of 95.0%. To provide the easy to use interface to doctors I have developed a website that will take the data and display the output with accuracy and time taken to predict.


## Languages or Frameworks Used 

  * Python: language
  * NumPy: library for numerical calculations
  * Pandas: library for data manipulation and analysis
  * SkLearn: library which features various classification, regression and clustering algorithms
  * Flask: microframework for building web applications using Python.
  
## Project Setup
  
  * First Clone the repository.
  * Create the virtual environment for the project.
  * Install the required packages using requirements.txt inside the environemnt using pip.
  * run the app.py as `python <app.py>`
  * Web Application will be hosted at  `127.0.0.1:5000`
  * Enter the URL in the browser Application will be hosted.
  * Enter the details of the tumor to detect the type of the cancer with more than 95% accuracy.
  
<hr style="border:2px solid gray"> </hr>

 ![Home Page](static/images/homepage1.png)

<hr style="border:2px solid gray"> </hr>

 ![Home Page](static/images/HomePage2.png)

<hr style="border:2px solid gray"> </hr>

  ![Home Page](static/images/homepage3.png)

<hr style="border:2px solid gray"> </hr>

![Home Page](static/images/predict.png)