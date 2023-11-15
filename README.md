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
![243253227-2ee73eb9-9122-4bbe-a23c-b3dc6b77e870](https://github.com/BaskaranV15/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118703522/6856b9a5-f71a-4aa3-a629-56746f247eb5)

![243253232-f881ca6c-0838-41c6-ab4d-53e38016911e](https://github.com/BaskaranV15/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118703522/71a90716-976f-4b5b-8e18-669cb393d2fc)

![243253241-bc457dd4-2cf4-4bc9-a57f-f31370925bf5](https://github.com/BaskaranV15/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118703522/cf6faeb7-7ee1-41cf-abd7-99a678b17cc2)

![243253264-45b6bf1f-f929-417c-9239-bcb82827824c](https://github.com/BaskaranV15/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118703522/b2d3d3c2-2288-4fa2-b3d7-7d3836d663f4)

![243253304-c7d01baa-d8ff-4bde-9904-4bbcfcad44b6](https://github.com/BaskaranV15/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118703522/f997294e-5434-4610-851d-e92a59aab15b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
