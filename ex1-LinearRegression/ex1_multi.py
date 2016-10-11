#!/usr/bin/env python3

import numpy as np

from feature_normalize import feature_normalize
from gradient_descent_multi import gradient_descent_multi


#############################
#        DATA LOADING       #
#############################
data = np.loadtxt('ex1data2.txt', delimiter=',')

X = data[:, :2]  # Extract the first two columns (features).
y = data[:,  2]  # Extract the last column (target values).
m = y.size       # Number of training samples.


#############################
#   FEATURE NORMALIZATION   #
#############################
print('Normalizing features...')
X, mu, sigma = feature_normalize(X)
print('  mu:     {}'.format(mu))
print('  sigma:  {}'.format(sigma))
print()


#############################
#   CONCATENATE X0 COLUMN   #
#############################
X = np.c_[np.ones((m, 1)), X]  # Columns of 1 for linear algebra purposes.


#############################
#      GRADIENT DESCENT     #
#############################
print('Running gradient descent...')

# Parameters:
alpha = 0.01
iterations = 1500

# Calculate theta through gradient descent:
theta = np.zeros(3)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, iterations)
print('  Theta after gradient descent: {}'.format(theta))
print()


#############################
#      TEST PREDICTION      #
#############################
sqft = 3890
rooms = 3

# Normalize and augment features with x0 = 1:
features = feature_normalize(np.array([[sqft, rooms]]), mu, sigma)[0]
features = np.c_[[1], features]

price_prediction = features.dot(theta)[0]

print('Predicted price for a house with 1650 sq ft and 3 rooms: {}'.format(
    price_prediction))
