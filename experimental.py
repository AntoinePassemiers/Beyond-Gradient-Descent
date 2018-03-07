import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def gradient_descent(X, A, f, alpha=.001):
    while True:
        y = f(A, X)
        grad = 2 * A * y
        A -= alpha * grad
        yield y


def custom_bfgs(X, A, f):
    B = None
    while True:
        y = f(A, X)

        grad = 2 * A * y
        B = 2 * np.outer(A, A) + 2 * y
        p = spsolve(B, -grad)
        # Line search... kind of
        alphas = [.00001, .0001, .001, .01, .1]
        values = np.asarray([f(A + a*p, X) for a in alphas])
        alpha = alphas[np.argmin(values)]
        # Update weights
        s = alpha * p
        A += s
        y_p = f(A, X)
        grad_p = 2 * A * y_p
        y_k = grad_p - grad
        yield y


n_samples = 1
n_variables = 500

X = np.random.rand(n_samples, n_variables) - (np.random.rand(n_variables) - 0.5) * 4
f = lambda a, x: np.dot(x, a) ** 2

X = np.squeeze(X)

A = (np.random.rand(n_variables) - 0.5) * 2

i = 0
for y1, y2 in zip(gradient_descent(X, np.copy(A), f), custom_bfgs(X, np.copy(A), f)):
    if i % 5 == 0:
        print("First order: %f - Second order: %f" % (y1, y2))
    i += 1

    if i == 80:
        break