# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate. 

4.Plot the Cost function using Gradient Descent and generate the required graph. 
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: JANARTHANAN B
RegisterNumber:  212223100014
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate = 0.1,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):                    
        #calculate predictions
        predictions = (x).dot(theta).reshape(-1,1)
                     
        #calculate errors
        errors = (predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-= learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta

data = pd.read_csv("C:/Users/ANANDAN S/Documents/ML labs/50_Startups.csv")
data.head()
#Assuming the last column is your target variable y
x= (data.iloc[1:,:-2].values)
x1 = x.astype(float)
scaler = StandardScaler()
y =(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled = scaler.fit_transform(x1)
y1_scaled = scaler.fit_transform(y)
print(x)
print(x1_scaled)
#learn model parameters
theta = linear_regression(x1_scaled,y1_scaled)
#predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_scaled),theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"prediction value: {pre}")
```

## Output:
![Screenshot 2024-08-30 143952](https://github.com/user-attachments/assets/97fc82d8-bbbf-4953-8a9e-329d834f2d84)


![Screenshot 2024-08-30 143921](https://github.com/user-attachments/assets/0ed7e9b7-ff7e-47f6-b045-623e071af9a2)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

