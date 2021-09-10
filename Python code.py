#Import Data Analysis Libraries
import pandas as pd  #for data I/O and dataframe functions
import numpy as np   #for linear Algebra 

#Importing Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline   #to show plots/graphs

#Getting data and doing Feature Engineering to understand data
loan= pd.read_csv('LendingClub\Loan_data.csv')
loan.head() 	#to check data/features in dataframe

loan.info() 	#to know the number of rows/columns and other details

loan.describe() #to get some statistical values

loan.shape  	#checking the shape of dataframe
loan.isnull()  	#checking for null values

loan[loan.isnull()].count()

##Exploratory Data Analysis

#Creating a histogram of two FICO distributions on top of each other, one for each credit.policy outcome
#using pandas built-in plots
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
loan[loan['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loan[loan['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

#analyzing the same by not.fully.paid column
plt.figure(figsize=(10,6))
loan[loan['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loan[loan['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

#Checking the counts of loans by purpose, with the color hue defined by not.fully.paid.
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loan,palette='Set1')

#checking the trend between FICO score and interest rate
sns.jointplot(x='fico',y='int.rate',data=loan,color='purple')

#checking the same as above but with more colums i.e. credit.policy and not.fully.paid 
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loan,hue='credit.policy',
           col='not.fully.paid',palette='Set1')

##Setting up the Data

loan.info()

#Categorical features
cat_feats = ['purpose']
final_data = pd.get_dummies(loan,columns=cat_feats,drop_first=True)
final_data.info()
final_data.head()

##Splitting the Features and Target variables
X = final_data.drop('not.fully.paid',axis=1) # Selecting All columns except not.fully.paid column
y = final_data['not.fully.paid']             # Selecting only Class column

X.shape  #Shape of X
y.shape  #Shape of y

##Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

X_train.shape  #Shape of X_train
X_test.shape   #Shape of X_test
y_train.shape  #Shape of y_train
y_test.shape   #Shape of y_test

##Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

#Creating an instance of DecisionTreeClassifier() called dtree and fitting it to the training data.
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

##Predictions and Evaluation of Decision Tree
#Creating predictions from the test set and creating a classification report and a confusion matrix.
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

##Training the Random Forest model
#Creating an instance of the RandomForestClassifier and fitting it to our training data from the previous step.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)

#Here n_estimators is the number of trees you want to build before taking the maximum voting or averages of predictions
rfc.fit(X_train,y_train)

##Predictions and Evaluation
#predicting the y_test values and evaluating the model
predictions = rfc.predict(X_test)

#creating a classification report and a confusion matrix for predictions
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))