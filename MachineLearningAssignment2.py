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
    
    # Split the dataset into labels and features
    # This seems to create a new dataframe of each column except label
    feature_vectors = product_df.loc[:, product_df.columns != 'label']
    labels = product_df['label']
    # print(feature_vectors.columns)
    # print(labels.head)
    
    # Print the number of sneakers
    print("The number of sneakers:", len(labels[labels == 0]))
    # Print the number of ankle boots
    print("The number of ankle boot:", len(labels[labels == 1]))
    
    # This is an array of pixal values
    # print(feature_vectors.values[0])
    

    # Example Sneaker
    plt.figure(1, figsize=(3, 3))
    plt.imshow(np.reshape(feature_vectors.values[3], (28, 28)), cmap='gray', interpolation='nearest')
    plt.show()
    
    # Example Ankle Boot
    plt.figure(1, figsize=(3, 3))
    plt.imshow(np.reshape(feature_vectors.values[0], (28, 28)), cmap='gray', interpolation='nearest')
    plt.show()
    
    
def main():
    preprocess()    
    
main()
