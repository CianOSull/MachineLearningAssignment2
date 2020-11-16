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

def preprocess():
    product_df = pd.read_csv("product_images.csv")
    print(product_df.head())


def main():
    preprocess()
    
    
main()
