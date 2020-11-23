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

def test(): 
    product_df = pd.read_csv("product_images.csv")

    # Adding .values at the end as it then returns a numpy array
    feature_vectors = product_df.loc[:, product_df.columns != 'label'].values
    labels = product_df['label'].values
    
    # Digit target:  [0 1 2 ... 8 9 8]
    # Digit data:  
    # [[ 0.  0.  5. ...  0.  0.  0.]
    #  [ 0.  0.  0. ... 10.  0.  0.]
    #  [ 0.  0.  0. ... 16.  9.  0.]
    #  ...
    #  [ 0.  0.  1. ...  6.  0.  0.]
    #  [ 0.  0.  2. ... 12.  0.  0.]
    #  [ 0.  0. 10. ... 12.  1.  0.]]
    
    # print(feature_vectors.values)
    print(feature_vectors)
    print(labels)
    # print("="*50)
    # print(feature_vectors.values[3])
    # print("="*50)
    # print(len(feature_vectors.values[3]))
    # print("="*50)
    # print(np.reshape(feature_vectors.values[3], (28, 28)))
    
    # train_data = feature_vectors[0:int(0.8*len(feature_vectors))]
    # train_target = labels[0:int(0.8*len(labels))]
    # test_data = feature_vectors[int(0.8*len(feature_vectors)):len(feature_vectors)]
    # test_target = labels[int(0.8*len(labels)):len(labels)]
    
    # train_data = feature_vectors.values[0:int(0.8*len(feature_vectors.values))]
    
    # train_target = digits.target[0:int(0.8*len(digits.target))]
    
    # test_data = feature_vectors.values[int(0.8*len(feature_vectors.values)):len(feature_vectors.values)]
    
    # test_target = digits.target[int(0.8*len(digits.target)):len(digits.target)]
    
    
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
    
     # Adding .values at the end as it then returns a numpy array
    feature_vectors = product_df.loc[:, product_df.columns != 'label'].values
    labels = product_df['label'].values
    
    # print(feature_vectors.columns)
    # print(labels.head)
    
    # Print the number of sneakers
    print("The number of sneakers:", len(labels[labels == 0]))
    # Print the number of ankle boots
    print("The number of ankle boot:", len(labels[labels == 1]))
    
    # This is an array of pixal values
    # print(feature_vectors.values[0])
    

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
    kf = model_selection.KFold(n_splits=2, shuffle=True)
    
    results = []
    
    # for train_index,test_index in kf.split(train_data):
    # clf1 = linear_model.Perceptron()
    # clf2 = svm.SVC(kernel="rbf", gamma=1e-3)    
    # clf3 = svm.SVC(kernel="sigmoid", gamma=1e-4)    

    # clf1.fit(train_data[train_index], train_target[train_index ])
    # prediction1 = clf1.predict(train_data[test_index])

    # clf2.fit(train_data[train_index], train_target[train_index])
    # prediction2 = clf2.predict(train_data[test_index])

    # clf3.fit(train_data[train_index], train_target[train_index])
    # prediction3 = clf3.predict(train_data[test_index])
    
    # score1 = metrics.accuracy_score(train_target[test_index], prediction1)
    # score2 = metrics.accuracy_score(train_target[test_index], prediction2)
    # score3 = metrics.accuracy_score(train_target[test_index], prediction3)
    
    # print("Perceptron accuracy score: ", score1)
    # print("SVM with RBF kernel accuracy score: ", score2)
    # print("SVM with Sigmoid kernel accuracy score: ", score3)
    # print()

    # if score1<best_score:
    #     best_clf = clf1
    # if score2<best_score:
    #     best_clf = clf2
    # if score3<best_score:
    #     best_clf = clf3
    
    for train_index,test_index in kf.split(train_data):
        perceptron = linear_model.Perceptron()
       
        # print(train_index)
        print("====================")
        # print(test_index)
        
        
        # clf1.fit(train_data[train_index], train_target[train_index ])
        # prediction1 = clf1.predict(train_data[test_index])
        perceptron.fit(train_data[train_index], train_target[train_index])
        prediction = perceptron.predict(train_data[test_index])
        print(prediction)
        break
    
    #     score = metrics.accuracy_score(train_target[test_index], prediction)
    #     results.append(score)
    
    # # prediction = perceptron.predict(test_data)
    # print(results)
        
def main():
    train_data, train_target, test_data, test_target = preprocess()    
    
    # print(train_target)
    
    perceptron(train_data, train_target, test_data, test_target)
    # test()
    
main()
