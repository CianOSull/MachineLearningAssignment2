# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:20:05 2020

@author: Cian
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model
from sklearn import svm

# Task 1
def preprocess():
    print("======================Task1======================")
    
    product_df = pd.read_csv("product_images.csv")
    # take a look at the dataset
    # print(product_df.head())
    # Take a look at the columns
    # print(product_df.columns)
    # Take a look at the labels
    # print(product_df['label'].head())
    
    # print(product_df.columns)    
    
    # Adding .values at the end as it then returns a numpy array
    feature_vectors = product_df.loc[:, product_df.columns != 'label'].values
    labels = product_df['label'].values
    
    # Print the number of sneakers
    print("The number of sneakers:", len(labels[labels == 0]))
    # Print the number of ankle boots
    print("The number of ankle boot:", len(labels[labels == 1]))
    

    # Example Sneaker
    plt.figure(1, figsize=(3, 3))
    plt.imshow(np.reshape(feature_vectors[3], (28, 28)), cmap='gray', interpolation='nearest')
    plt.show()
    
    # Example Ankle Boot
    plt.figure(1, figsize=(3, 3))
    plt.imshow(np.reshape(feature_vectors[0], (28, 28)), cmap='gray', interpolation='nearest')
    plt.show()
     
    
    # Sample variable for the amount of samples form the feature vector to use
    no_samples = 0.8
    
    
    # Starting off with reducted amount
    train_data = feature_vectors[0:int(no_samples*len(feature_vectors))]
    train_target = labels[0:int(no_samples*len(labels))]
    test_data = feature_vectors[int(no_samples*len(feature_vectors)):len(feature_vectors)]
    test_target = labels[int(no_samples*len(labels)):len(labels)]
    
    # Returns train and test vars and the number of samples
    return train_data, train_target, test_data, test_target, no_samples*len(feature_vectors)


# Task 2
def perceptron(train_data, train_target, test_data, test_target, no_samples, no_splits):
    
    kf = model_selection.KFold(n_splits=no_splits, shuffle=True)
    
    training_times = []
    prediction_times = []
    
    best_score = 1e100
    
    print("======================Task2======================")
    
     # Create a k-fold cross validation procedure to split the data into training and evaluation subsets [1 point]. 
    for train_index,test_index in kf.split(train_data):
        print("====================Split====================")
        
        perceptron = linear_model.Perceptron()
        
        # Measure the processing time required for training [1 point], 
        # This will create a time variable from this point
        start = time.time()
        
        # Train a perceptron classifier on the training subsets [1 point]
        perceptron.fit(train_data[train_index], train_target[train_index])
        
        # This will stop counting time
        stop = time.time()
        
        # Absolute means that it will always be printed positively
        print("Training time:", abs(stop - start), "secs")
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
        
        print("Prediction time:", abs(stop - start), "secs")
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
            best_perceptron = perceptron
    
    # Break up the results from each split
    print("="*50)
    
    # Test the model on unseen data and get a score
    test_prediction = best_perceptron.predict(test_data)
    
    # and the prediction accuracy [1 point]. 
    print("Prediciton accuracy score:", metrics.accuracy_score(test_target, test_prediction))
    
    # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point] 
    print("Minimum training time per training sample:", min(training_times)/no_samples, "secs")
    print("Maximum training time per training sample:", max(training_times)/no_samples, "secs")
    print("Average training time per training sample:", (sum(training_times)/len(training_times))/no_samples, "secs")
    print("Minimum predicition time per evaulation sample:", min(prediction_times)/no_samples, "secs")
    print("Maximum predicition time per evaulation sample:", max(prediction_times)/no_samples, "secs")
    print("Average predicition time per evaulation sample:", (sum(prediction_times)/len(prediction_times))/no_samples, "secs")
    
    # Confused by sample question, have it written in two ways with above being main and this one
    # being antoher attempt that switched from.
    # print("Minimum training time per training sample:", min(training_times), "secs")
    # print("Maximum training time per training sample:", max(training_times), "secs")
    # print("Average training time per training sample:", sum(training_times)/len(training_times), "secs")
    # print("Minimum predicition time per evaulation sample:", min(prediction_times), "secs")
    # print("Maximum predicition time per evaulation sample:", max(prediction_times), "secs")
    # print("Average predicition time per evaulation sample:", (sum(prediction_times)/len(prediction_times)), "secs")
       
    
# Task 3
def svm_func(train_data, train_target, test_data, test_target, no_samples, no_splits):     
    
   
    # for each split. Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point] 

    # Determine a good value for 𝛾 based on the mean accuracies you calculated [1 point]. 
    # Use a sufficient number of splits and vary the number of samples to observe the effect on runtime and accuracy [1 point].
    
    # ================ Currently Tested Gamma and Ksplits and accuracies: ================
    # Tests where no_samples was 0.08:
    # Ksplits:  Tested Gamma:   RBF Acc:                Linear Acc
    # 2         1e-4            0.49906832298136644     0.9427018633540373
    # 3         1e-4            0.49906832298136644     0.9427018633540373
    # 6         1e-4            0.49906832298136644     0.943944099378882
    # 2         1e-5            0.5001552795031056      0.9411490683229814
    # 2         1e-9            0.9059006211180124      0.940527950310559
    
    # Tests where no_samples was 0.8:
    # Ksplits:  Tested Gamma:   RBF Acc:                Linear Acc
    # 2         1e-9            0.9239285714285714      0.935
    
    # A         A               A                       A
    
    # Create a k-fold cross validation procedure to split the data into training and evaluation subsets [1 point]. 
    kf = model_selection.KFold(n_splits=no_splits, shuffle=True)
    
    linear_training_times = []
    linear_prediction_times = []
    
    rbf_training_times = []
    rbf_prediction_times = []
    rbf_accuracies = []
    
    best_score_linear = 1e100
    best_score_rbf = 1e100
    
    print("======================Task3======================")
    
    for train_index,test_index in kf.split(train_data):
        print("====================Split====================")
        
        # Linear SVM information will  go from below
        print("====================Linear Kernal====================")
        
        # Train a support vector machine classifier on the training subsets. Try a linear kernel [1 point] 
        linear = svm.SVC(kernel="linear")    
        
        # Measure the time required for training [1 point], 
        # Start Training Time
        start = time.time()
        
        linear.fit(train_data[train_index], train_target[train_index])
        
        # Stop Training Time
        stop = time.time()
        
        print("Training time:", abs(stop - start), "secs")
        linear_training_times.append(abs(stop - start))
        
        # the time required for prediction [1 point], and
        # Start Prediciton Time
        start = time.time()
        
        predictionLinear = linear.predict(train_data[test_index])
        
        # Stop Prediciton Time
        stop = time.time()
        
        print("Prediction time:", abs(stop - start), "secs")
        linear_prediction_times.append(abs(stop - start))
        
        # determine the accuracy score of the classification [1 point] 
        linear_score = metrics.accuracy_score(train_target[test_index], predictionLinear)
        print("Linear Accuracy Score: ", linear_score)
        
        # and the confusion matrix [1 point] 
        cLinear = metrics.confusion_matrix(train_target[test_index], predictionLinear)
        
        # Setting these to varaibles makes them easeir to read i think
        true_sneakers = cLinear[0,0]
        true_ankleboots = cLinear[1,1]            
        false_sneakers = cLinear[1,0]
        false_ankleboots = cLinear[0,1]

        # Confusion matrix for linear.
        print("Linear True sneakers:", np.sum(true_sneakers))
        print("Linear True ankle boots:", np.sum(true_ankleboots))
        print("Linear False sneakers:", np.sum(false_sneakers))
        print("Linear False ankle boots:", np.sum(false_ankleboots))
        
        #==============================================================
        
        # Linear SVM information will  go from below
        print("====================RBF Kernal====================")
        
        # and a radial basis function kernel for different choices of the parameter 𝛾 [2 points].
        rbf = svm.SVC(kernel="rbf", gamma=1e-9) 
        
        # Measure the time required for training [1 point], 
        # Start Training Time
        start = time.time()        
        
        rbf.fit(train_data[train_index], train_target[train_index])
        
        # Stop Training Time
        stop = time.time()
        
        print("Training time:", abs(stop - start), "secs")
        rbf_training_times.append(abs(stop - start))
        
        # the time required for prediction [1 point], and
        # Start Prediciton Time
        start = time.time()
        
        predictionRBF = rbf.predict(train_data[test_index])
        
         # Stop Prediciton Time
        stop = time.time()
        
        print("Prediction time:", abs(stop - start), "secs")
        rbf_prediction_times.append(abs(stop - start))
        
        # determine the accuracy score of the classification [1 point] 
        rbf_score = metrics.accuracy_score(train_target[test_index], predictionRBF)
        print("RBF Accuracy Score: ", rbf_score)
        rbf_accuracies.append(rbf_score)
        
        # and the confusion matrix [1 point] 
        cRBF = metrics.confusion_matrix(train_target[test_index], predictionRBF)
        
        # Setting these to varaibles makes them easeir to read i think
        true_sneakers = cRBF[0,0]
        true_ankleboots = cRBF[1,1]            
        false_sneakers = cRBF[1,0]
        false_ankleboots = cRBF[0,1]

        # Confusion matrix for rbf.
        print("RBF True sneakers:", np.sum(true_sneakers))
        print("RBF True ankle boots:", np.sum(true_ankleboots))
        print("RBF False sneakers:", np.sum(false_sneakers))
        print("RBF False ankle boots:", np.sum(false_ankleboots))
        
        # Break up Results from each split
        # print("="*100)
        
        # Whichever kfold split has the best accuracy save it
        if  linear_score <  best_score_linear:
            best_linear = linear
            
        if  rbf_score <  best_score_rbf:
            best_rbf = rbf
        
        
    # and the prediction accuracy [1 point]. 
    print("="*50)
    
    print("Prediciton accuracy score for Linear:", metrics.accuracy_score(test_target, best_linear.predict(test_data)))
    
    
    # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point]     
    print("Minimum Linear training time per training sample:", min(linear_training_times)/no_samples, "secs")
    print("Maximum Linear training time per training sample:", max(linear_training_times)/no_samples, "secs")
    print("Average Linear training time per training sample:", (sum(linear_training_times)/len(linear_training_times))/no_samples, "secs")
    print("Minimum Linear predicition time per evaulation sample:", min(linear_prediction_times)/no_samples, "secs")
    print("Maximum Linear predicition time per evaulation sample:", max(linear_prediction_times)/no_samples, "secs")
    print("Average Linear predicition time per evaulation sample:", (sum(linear_prediction_times)/len(linear_prediction_times))/no_samples, "secs")
    
    print("="*50)
    
    print("Prediciton accuracy score for rbf:", metrics.accuracy_score(test_target, best_rbf.predict(test_data)))
    print("Mean Accuarcy of rbf accross splits:", sum(rbf_accuracies)/len(rbf_accuracies))
    
    # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # the prediction time per evaluation sample [1 point]     
    print("Minimum rbf training time per training sample:", min(rbf_training_times)/no_samples, "secs")
    print("Maximum rbf training time per training sample:", max(rbf_training_times)/no_samples, "secs")
    print("Average rbf training time per training sample:", (sum(rbf_training_times)/len(rbf_training_times))/no_samples, "secs")
    print("Minimum rbf predicition time per evaulation sample:", min(rbf_prediction_times)/no_samples, "secs")
    print("Maximum rbf predicition time per evaulation sample:", max(rbf_prediction_times)/no_samples, "secs")
    print("Average rbf predicition time per evaulation sample:", (sum(rbf_prediction_times)/len(rbf_prediction_times))/no_samples, "secs")
    
    # # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # # the prediction time per evaluation sample [1 point]     
    # print("Minimum Linear training time per training sample:", min(linear_training_times), "secs")
    # print("Maximum Linear training time per training sample:", max(linear_training_times), "secs")
    # print("Average Linear training time per training sample:", sum(linear_training_times)/len(linear_training_times), "secs")
    # print("Minimum Linear predicition time per evaulation sample:", min(linear_prediction_times)/no_samples, "secs")
    # print("Maximum Linear predicition time per evaulation sample:", max(linear_prediction_times)/no_samples, "secs")
    # print("Average Linear predicition time per evaulation sample:", sum(linear_prediction_times)/len(linear_prediction_times), "secs")
    
    # print("="*50)
    
    # print("Prediciton accuracy score for rbf:", metrics.accuracy_score(test_target, best_rbf.predict(test_data)))
    
    # # Calculate the minimum, the maximum, and the average of the training time per training sample [1 point], 
    # # the prediction time per evaluation sample [1 point]     
    # print("Minimum rbf training time per training sample:", min(rbf_training_times), "secs")
    # print("Maximum rbf training time per training sample:", max(rbf_training_times), "secs")
    # print("Average rbf training time per training sample:", (sum(rbf_training_times)/len(rbf_training_times)), "secs")
    # print("Minimum rbf predicition time per evaulation sample:", min(rbf_prediction_times), "secs")
    # print("Maximum rbf predicition time per evaulation sample:", max(rbf_prediction_times), "secs")
    # print("Average rbf predicition time per evaulation sample:", sum(rbf_prediction_times)/len(rbf_prediction_times), "secs")
    
    
def main():
    train_data, train_target, test_data, test_target, no_samples = preprocess()    
    
    # perceptron(train_data, train_target, test_data, test_target, no_samples, 2)
    
    svm_func(train_data, train_target, test_data, test_target, no_samples, 2)
    
main()
