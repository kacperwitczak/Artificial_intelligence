import matplotlib.pyplot as plt
import numpy as np

from data import get_data, inspect_data, split_data
from functions import (z_score, calculate_batch_gradient_descent, closed_form_solution, mean_squared_error,
                       transform_theta, linear_function)

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)
# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data.drop(columns=['MPG']).to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data.drop(columns=['MPG']).to_numpy()

x_observer_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_observer_test = np.c_[np.ones(x_test.shape[0]), x_test]
y_train_column = np.expand_dims(y_train, axis=1)

# TODO: calculate closed-form solution
theta_best_column = closed_form_solution(x_observer_train, y_train_column)
theta_best = theta_best_column.flatten()

# TODO: calculate error
print("Closed-form solution MSE for train set: ", mean_squared_error(x_observer_train, theta_best, y_train))
print("Closed-form solution MSE for test set: ", mean_squared_error(x_observer_test, theta_best, y_test))
print("Best theta: ", theta_best)

# Plot the results
indices = np.arange(len(y_test))
bar_width = 0.35

plt.bar(indices, y_test, bar_width, label='Actual', color='b')
plt.bar(indices + bar_width, linear_function(x_observer_test, theta_best), bar_width, label='Predicted', color='r')
plt.xlabel('Samples')
plt.ylabel('MPG')
plt.title('Closed form solution')
plt.legend()
plt.show()

# TODO: standardization
x_train_normalized = z_score(x_train)
y_train_normalized = z_score(y_train_column)
x_observer_train_normalized = np.c_[np.ones(x_train_normalized.shape[0]), x_train_normalized]


# TODO: calculate theta using Batch Gradient Descent
theta = np.zeros((x_observer_train_normalized.shape[1], 1))

theta_gradient_column = calculate_batch_gradient_descent(0.0001, theta, x_observer_train_normalized, y_train_normalized)
theta_gradient = theta_gradient_column.flatten()

mean_x = np.mean(x_train, axis=0)
std_x = np.std(x_train, axis=0)
mean_y = np.mean(y_train, axis=0)
std_y = np.std(y_train, axis=0)

transformed_theta_gradient = transform_theta(theta_gradient, mean_x, std_x, mean_y, std_y)

# TODO: calculate error
print("Batch Gradient Descent solution MSE for train set: ", mean_squared_error(x_observer_train, transformed_theta_gradient, y_train))
print("Batch Gradient Descent solution MSE for test set: ", mean_squared_error(x_observer_test, transformed_theta_gradient, y_test))
print("Batch Gradient Descent theta: ", transformed_theta_gradient)

# plot the regression
indices = np.arange(len(y_test))
bar_width = 0.35

plt.bar(indices, y_test, bar_width, label='Actual', color='b')
plt.bar(indices + bar_width, linear_function(x_observer_test, transformed_theta_gradient), bar_width, label='Predicted', color='r')
plt.xlabel('Samples')
plt.ylabel('MPG')
plt.title('Gradient Descent Solution')
plt.legend()
plt.show()