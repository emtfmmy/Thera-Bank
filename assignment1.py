# import statements 
from lightgbm import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os
#from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Filename of the dataset to use for training and validation
train_data = "Bank_Personal_Loan_Train1.csv"
# Filename of test dataset to apply your model and predict outcomes 
test_data = "Bank_Personal_Loan_Test1.csv"


# Load the trainig data, clean/prepare and obtain training and target vectors, 
def load_prepare():
    df = pd.read_csv(train_data)
    X = df.drop('PersonalLoan', axis=1, inplace=False)
    y = df['PersonalLoan'] 
    # return training vector and target vector
    return X, y

# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_1(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, ['Age', 'Experience', 'Income', 'Family', 'CCAvg']),
        ('cat', OneHotEncoder(handle_unknown="ignore"), ['Education'])
    ])
    pipeline = Pipeline([
        ('ct', column_transformer),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X_train,y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict).round(4)
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline


# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, ['Age', 'Experience', 'Income', 'Family', 'CCAvg']),
        ('cat', OneHotEncoder(handle_unknown="ignore"), ['Education'])
    ])
    pipeline = Pipeline([
        ('ct', column_transformer),
        ('clf', KNeighborsClassifier(n_neighbors=3))
    ])
    pipeline.fit(X_train,y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict).round(4)
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline


# This your final and improved model pipeline
# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
# Save your final pipeline to a file "pipeline.pkl"   
def build_pipeline_final(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, ['Age', 'Experience', 'Income', 'Family', 'CCAvg']),
        ('cat', OneHotEncoder(handle_unknown="ignore"), ['Education'])
    ])
    pipeline = Pipeline([
        ('ct', column_transformer),
        ('rf', RandomForestClassifier(n_estimators=10))
    ])
    # fit pipeline (train)
    pipeline.fit(X_train, y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict).round(4)
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
    # your code goes here
    pickle.dump(pipeline, open('final_pipeline.pkl','wb'))
    # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
    return training_accuracy, confusion_matrix, pipeline


# Load final pipeline (pipe.pkl) and test dataset (test_data)
# Apply the pipeline to the test data and predict outcomes
def apply_pipeline():
    pipeline = pickle.load(open('final_pipeline.pkl','rb'))
    df = pd.read_csv(test_data)
    df['PersonalLoan'] = pipeline.predict(df)
    predictions = df['PersonalLoan']
    # return predictions or outcomes
    return predictions

