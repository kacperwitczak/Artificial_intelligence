import numpy as np


def linear_function(X, theta):
    return np.dot(X, theta)


def mean_squared_error(X, theta, y):
    return np.mean((linear_function(X, theta) - y) ** 2)


def z_score(X):
    mean_values = np.mean(X, axis=0)
    stddev_values = np.std(X, axis=0)

    z_scores = (X - mean_values) / stddev_values
    return z_scores


def calculate_gradient_mse(X, theta, y):
    m = X.shape[0]
    g = (2/m)*(X.T @ (X @ theta - y))
    return g


def calculate_batch_gradient_descent(LR, theta, X, y, epsilon=1e-6):
    prev_cost = 0
    while True:
        g = calculate_gradient_mse(X, theta, y)
        theta = theta - LR * g

        cost = mean_squared_error(X, theta, y)

        if abs(cost - prev_cost) <= epsilon:
            break
        prev_cost = cost

    return theta


def closed_form_solution(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def transform_theta(theta, mean_x, std_x, mean_y, std_y):
    transformed_theta = np.copy(theta)

    transformed_theta[1:] = transformed_theta[1:] * (std_y / std_x)

    transformed_theta[0] = mean_y - np.sum(transformed_theta[1:] * mean_x)

    return transformed_theta
