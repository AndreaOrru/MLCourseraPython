#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import sklearn.linear_model

from feature_normalize import feature_normalize
from gradient_descent_multi import gradient_descent_multi

# Ignore errors about non-present drivers on OS X:
import warnings
warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')


#############################
#        DATA LOADING       #
#############################
data = np.loadtxt('ex1data2.txt', delimiter=',')

X = data[:, :2]  # Extract the first two columns (features).
y = data[:,  2]  # Extract the last column (target values).
m = y.size       # Number of training samples.

X_orig = X       # Save the original matrix.


#############################
#   FEATURE NORMALIZATION   #
#############################
print('NORMALIZING FEATURES...')
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
print('RUNNING GRADIENT DESCENT...')

# Parameters:
alpha = 0.01
iterations = 1500

# Calculate theta through gradient descent:
theta = np.zeros(3)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, iterations)
print('  Theta after gradient descent: {}'.format(theta))


#############################
#      TEST PREDICTION      #
#############################
area = 2000
rooms = 3

# Normalize and augment features with x0 = 1:
features = feature_normalize(np.array([[area, rooms]]), mu, sigma)[0]
features = np.c_[[1], features]

price_prediction = features.dot(theta)[0]

print('  Predicted price for a house with {} sq ft '
      'and {} rooms: ${:.2f}'.format(area, rooms, price_prediction))
print()


#############################
#      NORMAL EQUATIONS     #
#############################
print('RUNNING NORMAL EQUATIONS SOLVER...')

# X is our matrix of features augmented with x0 = 1:
norm_eq_theta = np.linalg.pinv((X.T).dot(X)).dot(X.T).dot(y)
print('  Theta after normal equations: {}'.format(norm_eq_theta))

# Predict price using normal equations:
norm_eq_price = X.dot(norm_eq_theta)[0]

print('  Predicted price (normal equation method) for a house with {} sq ft '
      'and {} rooms: ${:.2f}'.format(area, rooms, norm_eq_price))
print()


#############################
#    SKLEARN LINEAR MODEL   #
#############################
print('RUNNING SKLEARN LINEAR REGRESSION...')

# X_orig is our matrix of features WITHOUT the augmented x0 = 1 column:
sk_linear_regression = sklearn.linear_model.LinearRegression()
sk_linear_regression.fit(X_orig, y)

predictions = sk_linear_regression.predict([[area, rooms]])
sklearn_price = predictions[0]

print('  Predicted price (SKLearn linear regression) for a house with {} sq ft '
      'and {} rooms: ${:.2f}'.format(area, rooms, sklearn_price))
print()


#############################
#        DRAW PLOTS         #
#############################

# J cost / Number of iterations:
plt.plot(J_history, '-')
plt.xlabel('Number of iterations')
plt.ylabel('J cost')
plt.show()
