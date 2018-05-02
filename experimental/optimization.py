import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def gradient_descent(X, A, f):
    while True:
        y = f(A, X)
        grad = 2 * A * y

        # Line search... kind of
        alphas = [.00001, .0001, .001, .01, .1]
        values = np.asarray([f(A - a*grad, X) for a in alphas])
        alpha = alphas[np.argmin(values)]
        A -= alpha * grad
        yield y


def bfgs(X, A, f):
    B = None
    while True:
        y = f(A, X)

        grad = 2 * A * y
        B = 2 * np.outer(A, A) + 2 * y
        p = scipy.sparse.linalg.spsolve(B, -grad)
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


def inner_product(a, b, c):
    return np.repeat(np.dot(a, c), len(a)) * b


class History:

    def __init__(self, m):
        self.padding = 0
        self.m = m
        self.values = list()
    
    def __getitem__(self, key):
        return self.values[key-self.padding]
    
    def __setitem__(self, key, value):
        if key-self.padding == len(self.values):
            self.values.append(value)
            if len(self.values) > self.m:
                self.values = self.values[1:]
                self.padding += 1
        elif key-self.padding < len(self.values):
            self.values[key-self.padding] = value
    
    def __len__(self):
        return len(self.values) + self.padding



def l_bfgs(X, A, f, m=8):
    old_grad = None

    y = History(m)
    s = History(m)
    alpha = History(m)

    k = -1
    while True:
        pred = f(A, X)
        grad = 2 * A * pred

        if k >= m:
            q = np.copy(grad)
            for i in range(k-1, k-m-1, -1):
                rho_i = 1. / np.dot(s[i], y[i])
                q -= alpha[i] * y[i]

            z = inner_product(y[k-1] / np.dot(y[k-1], y[k-1]), s[k-1], q)

            for i in range(k-m, k, 1):
                rho_i = 1. / np.dot(s[i], y[i])
                beta = rho_i * np.dot(y[i], z)
                z += s[i] * (alpha[i] - beta)
        else:
            z = grad

        # Line search... kind of
        aas = [.00001, .0001, .001, .01, .1]
        values = np.asarray([f(A - a*z, X) for a in aas])
        aa = aas[np.argmin(values)]
        # Update weights
        ss = aa * z
        A -= ss
        yield pred

        if old_grad is not None:
            y[k] = grad - old_grad
            s[k] = ss
            rho_i = 1. / np.dot(s[k], y[k])
            alpha[k] = rho_i * np.dot(s[k], grad)

        old_grad = grad
        k += 1


def main():
    n_samples = 1
    n_variables = 500

    X = np.random.rand(n_samples, n_variables) - (np.random.rand(n_variables) - 0.5) * 4
    f = lambda a, x: np.dot(x, a) ** 2

    X = np.squeeze(X)

    A = (np.random.rand(n_variables) - 0.5) * 10

    i = 0
    for y1, y2 in zip(gradient_descent(X, np.copy(A), f), l_bfgs(X, np.copy(A), f)):
        if i % 5 == 0:
            print("First order: %f - Second order: %f" % (y1, y2))
        i += 1

        if i == 80:
            break


main()