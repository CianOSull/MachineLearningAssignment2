# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:20:05 2020

@author: Cian
"""
import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
import random

from sklearn import datasets
    
# Task 1
def preprocess():
    product_df = pd.read_csv("product_images.csv")
    # take a look at the dataset
    # print(product_df.head())
    # Take a look at the columns
    # print(product_df.columns)
    # Take a look at the labels
    # print(product_df['label'].head())
    
    # print(product_df.columns)    
    
    # Did a quick check for empty values and found none    
    # for col in product_df.columns:
    #     print(col, ": " , product_df[col].isnull().sum())
    
    # Split the dataset into labels and features
    # This seems to create a new dataframe of each column except label
    # feature_vectors = product_df.loc[:, product_df.columns != 'label']
    # labels = product_df['label']
    
    # Need features and target to be in this format of numpy array
    # Digit target:  [0 1 2 ... 8 9 8]
    # Digit data:  
    # [[ 0.  0.  5. ...  0.  0.  0.]
    #  [ 0.  0.  0. ... 10.  0.  0.]
    #  [ 0.  0.  0. ... 16.  9.  0.]
    #  ...
    #  [ 0.  0.  1. ...  6.  0.  0.]
    #  [ 0.  0.  2. ... 12.  0.  0.]
    #  [ 0.  0. 10. ... 12.  1.  0.]]
    
    # Adding .values at the end as it then returns a numpy array
    feature_vectors = product_df.loc[:, product_df.columns != 'label'].values
    labels = product_df['label'].values
    
    # Print the number of sneakers
    print("The number of sneakers:", len(labels[labels == 0]))
    # Print the number of ankle boots
    print("The number of ankle boot:", len(labels[labels == 1]))
    

    # Example Sneaker
    # plt.figure(1, figsize=(3, 3))
    # plt.imshow(np.reshape(feature_vectors.values[3], (28, 28)), cmap='gray', interpolation='nearest')
    # plt.show()
    plt.figure(1, figsize=(3, 3))
    plt.imshow(np.reshape(feature_vectors[3], (28, 28)), cmap='gray', interpolation='nearest')
    plt.show()
    
    # Example Ankle Boot
    # plt.figure(1, figsize=(3, 3))
    # plt.imshow(np.reshape(feature_vectors.values[0], (28, 28)), cmap='gray', interpolation='nearest')
    # plt.show()
    plt.figure(1, figsize=(3, 3))
    plt.imshow(np.reshape(feature_vectors[0], (28, 28)), cmap='gray', interpolation='nearest')
    plt.show()
     
    # The number of rows taht will be used as samples
    # Initially starting with 200 of each
    # no_samples = 200
    
    # # Samples of feature vector and labels
    # fv_samples = feature_vectors[:no_samples]
    # labels_samples = labels[no_samples]
    
    # Split the data into train and test so its parametised
    # train_data = feature_vectors[0:int(0.8*len(feature_vectors))]
    # train_target = labels[0:int(0.8*len(labels))]
    # test_data = feature_vectors[int(0.8*len(feature_vectors)):len(feature_vectors)]
    # test_target = labels[int(0.8*len(labels)):len(labels)]

    # Starting off with reducted amount
    train_data = feature_vectors[0:int(0.08*len(feature_vectors))]
    train_target = labels[0:int(0.08*len(labels))]
    test_data = feature_vectors[int(0.08*len(feature_vectors)):len(feature_vectors)]
    test_target = labels[int(0.08*len(labels)):len(labels)]
    
    return train_data, train_target, test_data, test_target

def perceptron(train_data, train_target, test_data, test_target):
   
   
    # Measure the processing time required for training [1 point], 
    # the processing time required for prediction [1 point], 
   
    
    # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point] 
    # 
    # Use a sufficient number of splits and vary the number of samples to
    # observe the effect on runtime and accuracy [1 point].
    
    kf = model_selection.KFold(n_splits=2, shuffle=True)

    best_score = 1e100
    
     # Create a k-fold cross validation procedure to split the data into training and evaluation subsets [1 point]. 
    for train_index,test_index in kf.split(train_data):
        perceptron = linear_model.Perceptron()

        print("="*100)
        
        # Train a perceptron classifier on the training subsets [1 point]
        perceptron.fit(train_data[train_index], train_target[train_index])
        
         # and predict labels for the evaluation subsets [1 point]. 
        prediction = perceptron.predict(train_data[test_index])
        
         # and determine the accuracy score of the classification [1 point] 
        score = metrics.accuracy_score(train_target[test_index], prediction)
        
        # and the confusion matrix [1 point] 
        C = metrics.confusion_matrix(train_target[test_index], prediction)
        
        # Setting these to varaibles makes them easeir to read i think
        true_sneakers = C[0,0]
        true_ankleboots = C[1,1]            
        false_sneakers = C[1,0]
        false_ankleboots = C[0,1]

        # for each split.
        print("Accuracy Score: ", score)
        print("True sneakers:", np.sum(true_sneakers))
        print("True ankle boots:", np.sum(true_ankleboots))
        print("False sneakers:", np.sum(false_sneakers))
        print("False ankle boots:", np.sum(false_ankleboots))
        
        # Whichever kfold split has the best accuracy save it
        if  score <  best_score:
            best_clf = perceptron
     
    # Test the model on unseen data and get a score
    test_prediction = best_clf.predict(test_data)
    # and the prediction accuracy [1 point]. 
    print("="*100)
    print("Prediciton accuracy score:", metrics.accuracy_score(test_target, test_prediction))
        
        
def main():
    train_data, train_target, test_data, test_target = preprocess()    
    
    
    perceptron(train_data, train_target, test_data, test_target)
    
main()
