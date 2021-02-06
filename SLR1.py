# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:24:00 2021

@author: shra1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'E:\CLASSES\15.6th Feb_Class15\15.6th Jan_Class15\SIMPLE LINEAR REGRESSION\Salary_Data.csv')
df.head()
X= df.iloc[:,:-1]
Y= df.iloc[:,1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
Regressor= LinearRegression()
Regressor.fit(X_train,Y_train)
Y_pred= Regressor.predict(X_test)
plt.scatter(X_train, Y_train,c='R')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test,Regressor.predict(X_test),c='g')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('year of exp')
plt.ylabel('sal')
plt.show()