# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AT ABHISHEK
RegisterNumber: 212224040006
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```
## Output:

<img width="377" height="417" alt="image" src="https://github.com/user-attachments/assets/4145f932-33b9-469b-8690-7e9293d99eb2" />

<img width="597" height="503" alt="image" src="https://github.com/user-attachments/assets/461a8903-9b0b-43c7-9f29-06149b4a16b0" />

<img width="258" height="411" alt="image" src="https://github.com/user-attachments/assets/e10d7806-ef5e-4565-8a16-1e45c6bf196f" />

<img width="227" height="115" alt="image" src="https://github.com/user-attachments/assets/bad0ee26-0915-4c5f-9f80-b67828a6a762" />

<img width="215" height="217" alt="image" src="https://github.com/user-attachments/assets/f53616ca-37bd-42a1-a1e0-4e5ca7583e70" />

<img width="1228" height="95" alt="image" src="https://github.com/user-attachments/assets/7b9be576-7ee9-4963-a119-44b11e052f57" />

<img width="637" height="527" alt="image" src="https://github.com/user-attachments/assets/c82861e2-6810-42eb-8505-f4784fe57fd4" />

<img width="232" height="182" alt="image" src="https://github.com/user-attachments/assets/d7f6670c-bc57-41a6-bcbf-965564430466" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
