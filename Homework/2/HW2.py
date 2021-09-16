# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:01:04 2021

@author: Ilan Valencius
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, model_selection


### Part 1: Univariate Linear Regression with Gradient Descent

# Define function to return gradient
def gradient(W, x, y):
    w0 = W[0]
    w1 = W[1]
    N = len(x)
    
    # Derivatives
    wrtw0 = lambda x,y: -2*(y-w0-w1*x)
    wrtw1 = lambda x,y: -2*x*(y-w0-w1*x)
    
    # Variables to hold sums
    sum_w0 = 0
    sum_w1 = 0
    
    # Iterate over list and perform summation
    for (xn, yn) in zip(x, y):
        sum_w0 += wrtw0(xn, yn)
        sum_w1 += wrtw1(xn, yn)
        
    # Divide by N to get final answer and store in tuple
    grad = (1/N*sum_w0, 1/N*sum_w1)
    
    return grad

#
        
# Organize function calls for part 1
def univariate():
    # Define parameters for learning
    eta = 0.1
    iterations = 40
    
    # Import csv file
    x = []
    y = []
    with open('unilinear.csv', newline='') as file:
        csv_in = csv.reader(file)
        for row in csv_in:
            x.append(row[0])
            y.append(row[1])
    
