import numpy as np
import matplotlib.pyplot as plt
 
from data import get_data, inspect_data, split_data
 
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
x_train = train_data['Weight'].to_numpy()
 
y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()
 
# TODO: calculate closed-form solution
def formulate_matrix(x_raw, y_raw):
    X = np.hstack([np.ones((x_raw.shape[0], 1)), x_raw.reshape(-1, 1)])
    y = y_raw.reshape(-1, 1)
    return X, y
 
X_train, Y_train = formulate_matrix(x_train, y_train)
 
 
# theta_best = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ Y_train)
theta_best = np.linalg.solve(X_train.T @ X_train, X_train.T @ Y_train)
 
# TODO: calculate error
def mse(Y_pred, Y):
    return np.mean((Y_pred - Y)**2)
 
Y_pred_train = X_train @ theta_best
mse_train = mse(Y_pred_train, Y_train)
 
 
# evaluation (on test data)
X_test, Y_test = formulate_matrix(x_test, y_test)
Y_pred_test = X_test @ theta_best
mse_test = mse(Y_pred_test, Y_test)
 
print(f"{mse_train=}, {mse_test=}")
 
 
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
 
# TODO: standardization
def standardize(x_raw):
    m = np.mean(x_raw)
    s = np.std(x_raw)
    return (x_raw - m)/s, m, s
 
x_train_norm, m, s = standardize(x_train)
X_train, Y_train = formulate_matrix(x_train_norm, y_train)
 
# TODO: calculate theta using Batch Gradient Descent
 
theta = np.random.randn(2, 1)  # initially random
lr = 0.01
 
N = X_train.shape[0]
 
for epoch in range(1000):
    Y_train_pred = X_train @ theta
    grad = 2/N * X_train.T @ (Y_train_pred - Y_train)
    theta -= lr * grad
    if epoch % 50 == 0:
        print(mse(Y_train_pred, Y_train))
 
 
 
# TODO: calculate error
 
x_test = (x_test-m)/s
 
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()