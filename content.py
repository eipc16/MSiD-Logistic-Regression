# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def logistic_cost_function(w, x_train, y_train):
    sigma = sigmoid(x_train @ w)
    cost = -np.sum(np.log(np.abs(sigma + y_train - 1))) / y_train.shape[0]
    grad = x_train.T.dot(sigma - y_train) / y_train.shape[0]
    return cost, grad

def gradient_descent(obj_fun, w0, epochs, eta):
    w = w0
    func_values = np.empty([epochs, 1])
    for i in range(epochs):
        cost_func = obj_fun(w)
        w = w - eta * obj_fun(w)[1]
        func_values[i] = obj_fun(w)[0]
    return w, func_values

def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    w = w0
    func_values = np.empty((epochs, 1))
    x_batches = np.vsplit(x_train, x_train.shape[0] / mini_batch)
    y_batches = np.vsplit(y_train, y_train.shape[0] / mini_batch)
    for k in range(epochs):
        for x, y in zip(x_batches, y_batches):
            w = w - eta * obj_fun(w, x, y)[1]
        func_values[k] = obj_fun(w, x_train, y_train)[0]
    return w, func_values

def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    val, grad = logistic_cost_function(w, x_train, y_train)
    return val + regularization_lambda / 2 * (w[1:,0].T @ w[1:,0]), grad + regularization_lambda * np.vstack([0, w[1:]])

def prediction(x, w, theta):
    return sigmoid(x @ w) >= theta

def f_measure(y_true, y_pred):
    DTP = 2 * np.sum(np.bitwise_and(y_true, y_pred))
    FP_and_FN = np.sum(np.bitwise_xor(y_true, y_pred))
    return DTP / (DTP + FP_and_FN)

def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    F_measures = np.zeros((len(lambdas), len(thetas)))
    best_measure = -1
    for i in range(len(lambdas)):
        obj_fun = lambda w, x_train, y_train: regularized_logistic_cost_function(w, x_train, y_train, lambdas[i])
        w, _ = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        for j in range(len(thetas)):
            F_measures[i, j] = f_measure(y_val, prediction(x_val, w, thetas[j]))
            if F_measures[i, j] > best_measure:
                best_measure = F_measures[i, j]
                best_lambda, best_theta, best_w = lambdas[i], thetas[j], w

    F_measures = np.array(F_measures).reshape(len(lambdas), len(thetas))
    return best_lambda, best_theta, best_w, F_measures
