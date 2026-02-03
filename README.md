# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries: Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

2.Define the Linear Regression Function: Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

3.Load and Preprocess the Data: Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

4.Perform Linear Regression: Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

5.Make Predictions on New Data: Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

6.Print the Predicted Value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SIBHIRAAJ R
RegisterNumber: 212224230268
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)
dataset

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
## Read the file and display
<img width="1440" height="499" alt="image" src="https://github.com/user-attachments/assets/7634bb1e-5945-4d60-8225-dcab0cf48c52" />
## Categorizing columns
<img width="1322" height="503" alt="image" src="https://github.com/user-attachments/assets/54e32ad1-acb3-4f11-b734-990d41002823" />

## Labelling columns and displaying dataset
<img width="430" height="588" alt="image" src="https://github.com/user-attachments/assets/21225615-c014-40cc-bbea-7949a0d01a0a" />

## Display dependent variable
<img width="1184" height="509" alt="image" src="https://github.com/user-attachments/assets/ddfe3eff-d95b-4462-8b76-dcb0c4fc9430" />

## Printing accuracy
<img width="889" height="224" alt="image" src="https://github.com/user-attachments/assets/5e745953-b366-4f3f-b811-729a8dc8ebb0" />

## Printing Y
<img width="965" height="155" alt="image" src="https://github.com/user-attachments/assets/8db65389-1b63-433e-92eb-a4721b9923a3" />

## Printing y_prednew
<img width="642" height="124" alt="image" src="https://github.com/user-attachments/assets/709cb29f-60fc-474b-a7b0-633d56bd9efe" />
<img width="706" height="121" alt="image" src="https://github.com/user-attachments/assets/3bed5582-12cb-4d5b-be12-53a0f419c262" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

