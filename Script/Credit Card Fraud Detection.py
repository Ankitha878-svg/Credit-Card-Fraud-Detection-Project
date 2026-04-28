#Credit Card Fraud Detection Project

#Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Loading the dataset to a Pandas DataFrame

credit_card_data = pd.read_csv(r"C:\Users\PI\Documents\credit card project\creditcard.csv")
print(credit_card_data)

#First 5 rows of the dataset

print(credit_card_data.head())

#Last 5 rows of the dataset

print(credit_card_data.tail())

#Dataset information

print(credit_card_data.info())

#Checking the no.of missing values in each column

print(credit_card_data.isnull().sum())

#Distribution of legit transactions & fraudulent transactions

#This dataset is highly unbalanced

#0 --> Legitimate (Normal) transaction
#1 --> Fraudulent transaction

print(credit_card_data['Class'].value_counts())

#“This line selects the Class column from the dataset and counts the number of occurrences of each value,
#helping us understand how many legitimate and fraudulent transactions are present.”

#Separating the data for analysis

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

#“This code separates the dataset into two parts—legitimate and fraudulent transactions—using conditional  
#filtering. Then, the shape function is used to find the number of rows and columns in each category.”

#Statistical measures of the data

print(legit.Amount.describe())

print(fraud.Amount.describe())

#“This code calculates statistical measures like mean, standard deviation, minimum, maximum, & quartiles 
#for the transaction amount of legitimate transactions, helping to understand their distribution.”

#Compare the values for both transactions

print(credit_card_data.groupby('Class').mean())

#“This code groups the dataset based on the Class column and calculates the mean of each feature 
#for both legitimate and fraudulent transactions, helping to compare their behavior and identify 
#differences.”

#Under-Sampling

#Build a sample dataset containing similar distribution of "normal transactions" & "fraudulent transactions"

#Number of fraudulent transactions --> 492

legit_sample = legit.sample(n=492)

#Concatenating two DataFrames(legit_sample + fraud)

new_dataset = pd.concat([legit_sample , fraud],axis = 0) 

print(new_dataset.head())

print(new_dataset.tail())

print(new_dataset['Class'].value_counts()) #To find no.of rows in both legit n fraud transactions

#“This code performs undersampling by randomly selecting 492 legitimate transactions using the sample() 
#function to match the number of fraudulent transactions. Then, both datasets are combined using 
#pd.concat() with axis = 0, which means the data is concatenated row-wise. This results in a balanced 
#dataset containing equal numbers of legitimate and fraudulent transactions, which helps improve the
#performance of the machine learning model.”

print(new_dataset.groupby('Class').mean())

#“It helps verify that after undersampling, both classes are equally represented, while still retaining 
#differences in feature values that allow the model to distinguish between fraud and legitimate 
#transactions.”

#Splitting the data into Features & Targets

X = new_dataset.drop(columns = 'Class')
Y = new_dataset['Class']

print(X)
print(Y)

#“We separate the dataset into features (X) and target variable (Y), where X contains all 
#input variables & Y contains the Class column representing fraud or legitimate transactions.”
# Machine learning needs: X --> input features , Y --> target output

#Split the data into Training data & testing data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#“In this step, I used the train_test_split function to divide the dataset into training and testing sets.
#The input features X and target variable Y are split such that test_size = 0.2 allocates 20% of the data
#for testing and the remaining 80% for training. The parameter stratify = Y ensures that both training 
#and testing datasets maintain the same proportion of legitimate and fraudulent transactions, 
#which is important for imbalanced datasets. The random_state = 2 is used to ensure reproducibility, 
#meaning the data split remains consistent across multiple runs. 
#The outputs X_train and Y_train are used to train the model, 
#while X_test and Y_test are used to evaluate its performance.”

#X_train  | Features for training |
#X_test   | Features for testing  |
#Y_train  | Labels for training   |
#Y_test   | Labels for testing    |

# Model Training

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter = 2000, solver='liblinear')

# Training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

#“In this step, I scaled the data using StandardScaler to bring all features to the same range. 
#Then I initialized a Logistic Regression model with sufficient iterations and trained it using the 
#fit() function on the training data to learn patterns for classifying transactions.”

#Model Evaluation

#Accuracy Score

#Accuracy on Training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy Score on Training Data : ",training_data_accuracy)

#“In this step, I evaluated the model on the training data by predicting outcomes using the trained model
#and calculating accuracy using accuracy_score. This shows how well the model has learned from the 
#training dataset.”

#Accuracy on Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy Score on Test Data : ",test_data_accuracy)

#“The model achieved around 95% accuracy on training data and 91% on test data, 
#indicating good learning and generalization with minimal overfitting.”
