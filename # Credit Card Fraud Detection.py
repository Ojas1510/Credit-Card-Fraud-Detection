#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# In[100]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Loading the Dataset

# In[101]:


card_df = pd.read_csv('creditcard.csv')


# In[136]:


card_df.head()
# Here the values shown in:
#Collum_1 is time of the day
#Collum_2 to Collum_28 are the credit card details which are converted in numerical values using PCA 
#Collum_29 is Amount
#Collum_30 is label, [0,1]


# In[103]:


card_df.tail()


# In[104]:


# Data info:
card_df.info()


# In[105]:


# Checking for the missing values
card_df.isnull().sum()


# In[106]:


# Examining the distribution legit transactions & fraudt transactions
card_df['Class'].value_counts()


# This Dataset is highly unblanced
# Here collum 31 describe the class:
# 0 --> Normal Transaction
# 1 --> fraudulent transaction

# In[107]:


# separating the data on basis of the class:
legit = card_df[card_df.Class == 0]
fraud = card_df[card_df.Class == 1]


# In[108]:


print(legit.shape)
print(fraud.shape)


# In[109]:


# Applying statistical measures to amount collum of the datasets
# Analyzing the mean values of amount in both type of transactions 

legit.Amount.describe()


# In[110]:


fraud.Amount.describe()


# In[111]:


# comparing the values of the both transactions
card_df.groupby('Class').mean()


# Under-Sampling
# 
# Building a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# Number of Fraudulent Transactions --> 492
# 
# Under-sampling involves reducing the number of instances in the majority class to balance it with the number of instances in the minority class. The goal is to create a more balanced dataset that allows the machine learning model to learn from both classes equally and improve its ability to predict the minority class.
# Here we will use random Under-sampling.

# In[112]:


legit_sample = legit.sample(n=492,random_state=42)


# Concatenating two DataFrames

# In[113]:


new_df= pd.concat([legit_sample, fraud], axis=0)
# Here axis is '0' which means the concatination will be done row-wise


# In[114]:


new_df.head()


# In[115]:


new_df.tail()


# In[116]:


new_df['Class'].value_counts()


# In[117]:


new_df.groupby('Class').mean()
# to find the nature of dataset haven't changed, wheather we go good sample or bad sample 


# Splitting the data into Features & Targets
# 
# X will contain features and Y will have class lables which will act as target

# In[118]:


X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']


# In[119]:


print(X)


# In[120]:


print(Y)


# Split the data into Training data & Testing Data

# In[121]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)


# In[122]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training
# 
# Applying Logistic Regression: because we generally use logistic regression for binary class classification

# In[123]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[124]:


# training the Model;
model.fit(X_train, Y_train)


# Model Evaluation
# 
# Accuracy Score

# In[125]:


#First we will find the accuracy score for training data inorder to find accuracy of our model for training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[126]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[127]:


# Now find accuracy score for test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[128]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# Confusion Matrix:

# In[129]:


y_pred = model.predict(X_test)
#print(y_pred)
Y_test1= Y_test.to_numpy()
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test1.reshape(len(Y_test1),1)),1))


# In[130]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
cm = confusion_matrix(Y_test1, y_pred)
print(cm)


# In[131]:


#Visualizing confusion matrix for a better view

from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# AUC-ROC Score

# In[132]:


roc_auc_score(Y_test1, y_pred)


# Overall metrics report of the logistic regression

# In[133]:


import sklearn.metrics as metrics
print(metrics.classification_report(Y_test1, y_pred))


# In[ ]:




