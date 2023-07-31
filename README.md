# Credit-Card-Fraud-Detection
This repository contains code for a credit card fraud detection system. <br />
The dataset used for this project is sourced from Kaggle, and it is intended to demonstrate the process of building a basic fraud detection model using logistic regression.
# Table of Contents
1. Introduction <br />
2. Requirements <br />
3. Setup <br />
4. Data Preparation <br />
5. Data Analysis <br />
6. Model Building <br /> 
7. Model Evaluation <br />
# Introduction
Credit card fraud is a significant concern for financial institutions and consumers alike. This project aims to develop a simple fraud detection model using logistic regression. We'll use a dataset from Kaggle that contains information about various credit card transactions, including labels for whether the transaction is fraudulent (1) or legitimate (0).

# Requirements
Python 3.x <br />
Pandas <br />
NumPy <br />
Scikit-learn <br />


# Setup
Install the required dependencies using pip install -r requirements.txt or conda install -r- requirements.txt

# Data Preparation
Download the credit card fraud dataset from Kaggle https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. <br />
The dataset is expected to have the following columns: <br />
Collum_1 is time of the day <br />
Collum_2 to Collum_28 are the credit card details which are converted in numerical values using PCA <br />
Collum_29 is Amount <br />
Collum_30 is label, [0,1] <br />
# Data Preprocessing
Missing Values Analysis: <br />
Performing a missing values analysis is crucial to ensure the dataset's completeness and identify any potential data imbalances. <br />
Read the dataset into a Pandas DataFrame. <br />
Check for missing values in each column and handle them appropriately, such as removing or imputing missing data. <br />
Statistical Measures: <br />
Calculate and compare the mean values for the amount of legitimate transactions and fraudulent transactions. <br />
Compute other relevant statistical measures such as median, standard deviation, minimum, and maximum values for both types of transactions. <br />

# Data Splitting
Split the dataset into two separate DataFrames - one for fraud transactions and one for legitimate transactions - based on the given labels [0, 1].  <br />
# Data Undersampling
Undersample the majority class (legitimate transactions) to balance the dataset.

# Model Building
Create target (y) and features (X) from the undersampled data. <br />
Split the data into training and testing sets. <br />

# Model Training and Evaluation
Apply logistic regression to train the model.  <br />
Logistic regression is a binary classification algorithm. It predicts the probability of an instance belonging to a particular class based on input features.  <br />
Calculate the accuracy of the model.  <br />
Evaluate the model using the confusion matrix.  <br />
Using the confusion matrix, we can calculate various evaluation metrics to assess the model's performance.  <br />




