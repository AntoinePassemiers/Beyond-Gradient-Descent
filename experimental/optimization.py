import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def gradient_descent(X, A, f, alpha=.001):
    while True:
        y = f(A, X)
        grad = 2 * A * y
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


def l_bfgs(X, A, f, m=2):
    old_grad = None
    w = 10
    ss, ys = list(), list()
    while True:
        y = f(A, X)

        grad = 2 * A * y

        if len(ss) == w:
            q = np.copy(grad)
            alphas = list()
            for s_i, y_i in zip(reversed(ss), reversed(ys)):
                rho_i = 1. / np.dot(s_i, y_i)
                alpha_i = rho_i * np.dot(s_i, q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            alphas = list(reversed(alphas))

            z = np.einsum('i,i,i->i', ys[-1] / np.dot(ys[-1], ys[-1]), ss[-1], q)

            for s_i, p_i, alpha_i in zip(ss, ys, alphas):
                rho_i = 1. / np.dot(s_i, p_i)
                beta_i = rho_i * np.dot(y_i, z)
                z = z + s_i * (alpha_i - beta_i)
        else:
            z = grad


        #alphas = [.00001, .0001, .001, .01, .1]
        #values = np.asarray([f(A - a*z, X) for a in alphas])
        #alpha = alphas[np.argmin(values)]
        alpha = .001
        # Update weights
        s = alpha * z
        A -= s
        yield y

        if old_grad is not None:
            ys.append(grad - old_grad)
            ss.append(s)
            if len(ss) > w:
                ss, ys = ss[1:], ys[1:]
        old_grad = grad


n_samples = 1
n_variables = 500

X = np.random.rand(n_samples, n_variables) - (np.random.rand(n_variables) - 0.5) * 4
f = lambda a, x: np.dot(x, a) ** 2

X = np.squeeze(X)

A = (np.random.rand(n_variables) - 0.5) * 2

i = 0
for y1, y2 in zip(gradient_descent(X, np.copy(A), f), l_bfgs(X, np.copy(A), f)):
    if i % 5 == 0:
        print("First order: %f - Second order: %f" % (y1, y2))
    i += 1

    if i == 40:
        break
