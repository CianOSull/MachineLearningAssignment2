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
import time
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
    
    # Sample variable for how much to take
    # samples tested: 0.08, 0.8
    no_samples = 0.8
    
    # Starting off with reducted amount
    train_data = feature_vectors[0:int(no_samples*len(feature_vectors))]
    train_target = labels[0:int(no_samples*len(labels))]
    test_data = feature_vectors[int(no_samples*len(feature_vectors)):len(feature_vectors)]
    test_target = labels[int(no_samples*len(labels)):len(labels)]
    
    # Returns train and test vars and the number of samples
    return train_data, train_target, test_data, test_target, no_samples*len(feature_vectors)

def perceptron(train_data, train_target, test_data, test_target, no_samples):
    # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point] 
    # NOTE: minimium, maximum and training timer probably refers to the number of splits so save times to a list
    # 
    # Use a sufficient number of splits and vary the number of samples to
    # observe the effect on runtime and accuracy [1 point].
    
    kf = model_selection.KFold(n_splits=2, shuffle=True)
    
    training_times = []
    prediction_times = []
    
    best_score = 1e100
    
     # Create a k-fold cross validation procedure to split the data into training and evaluation subsets [1 point]. 
    for train_index,test_index in kf.split(train_data):
        perceptron = linear_model.Perceptron()

        print("="*100)
        
        # Measure the processing time required for training [1 point], 
        # This will create a time variable from this point
        start = time.time()
        
        # Train a perceptron classifier on the training subsets [1 point]
        perceptron.fit(train_data[train_index], train_target[train_index])
        
        # This will stop counting time
        stop = time.time()
        
        # Absolute means that it will always be printed positively
        print("Training time:", abs(stop - start))
        # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point]
        training_times.append(abs(stop - start))
        # I have no idea how this print statement works but it jsut does
        # print(f"Training time: {stop - start}s")
        
         # the processing time required for prediction [1 point],
        start = time.time()
        
         # and predict labels for the evaluation subsets [1 point]. 
        prediction = perceptron.predict(train_data[test_index])
        
         # This will stop counting time
        stop = time.time()
        
        print("Prediction time:", abs(stop - start))
        # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point]
        prediction_times.append(abs(stop - start))
        # print(f"Prediction time: {stop - start}s")
        
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
        
    # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point]     
    print("Minimum training time per training sample:", min(training_times)/no_samples)
    print("Maximum training time per training sample:", max(training_times)/no_samples)
    print("Average training time per training sample:", (sum(training_times)/len(training_times))/no_samples)
    print("Minimum predicition time per evaulation sample:", min(prediction_times)/no_samples)
    print("Maximum predicition time per evaulation sample:", max(prediction_times)/no_samples)
    print("Average predicition time per evaulation sample:", (sum(prediction_times)/len(prediction_times))/no_samples)
      
def svm(train_data, train_target, test_data, test_target, no_samples):
    
    # Train a support vector machine classifier on the training subsets. Try a linear kernel [1 point] 
    # and a radial basis function kernel for different choices of the parameter 𝛾 [2 points]. 
    # Predict the labels for the evaluation subsets [1 point]. 
    # Measure the time required for training [1 point], 
    # the time required for prediction [1 point], and
    # determine the accuracy score of the classification [1 point] 
    # and the confusion matrix [1 point] 
    # for each split. Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point] 
    # and the prediction accuracy [1 point]. 
    # Determine a good value for 𝛾 based on the mean accuracies you calculated [1 point]. 
    # Use a sufficient number of splits and vary the number of samples to observe the effect on runtime and accuracy [1 point].
    
    # Create a k-fold cross validation procedure to split the data into training and evaluation subsets [1 point]. 
    kf = model_selection.KFold(n_splits=2, shuffle=True)

     
def main():
    train_data, train_target, test_data, test_target, no_samples = preprocess()    
    
    
    perceptron(train_data, train_target, test_data, test_target, no_samples)
    
main()
