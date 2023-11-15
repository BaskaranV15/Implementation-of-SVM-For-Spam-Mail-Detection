# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the necessary python packages using import statements.

Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

Split the dataset using train_test_split.

Calculate Y_Pred and accuracy.

Print all the outputs.

End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BASKARAN V
RegisterNumber:  212222230020
*/
```
```PYTHON
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result=chardet.detect(rawdata.read(100000))
result

df=pd.read_csv("/content/spam.csv",encoding='Windows-1252')
df

df.info()

df.isnull().sum()

x=df['v1'].values
y=df['v2'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
