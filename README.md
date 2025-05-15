# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data. 
3. Use modelselection and Countvectorizer to preditct the values
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAMYA S
RegisterNumber: 212224040268
*/
```
```
import chardet
file = "//content//spam.csv"
with open(file, 'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv("/content/spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v2"].values
y = data["v1"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
![SVM For Spam Mail Detection](sam.png)
![image](https://github.com/user-attachments/assets/52065cdb-b29e-47c2-aa96-80b0ce4fb936)
![image](https://github.com/user-attachments/assets/f6ac8979-d1bc-41ae-87fd-ce939e0aa82b)
![image](https://github.com/user-attachments/assets/b23925a6-23bf-4294-840e-bcd3a55c0f53)
![image](https://github.com/user-attachments/assets/9f467446-6877-4735-ab50-fa151d31cb6b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
