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

def compute_rho(s_i, y_i):
    den = np.dot(s_i, y_i)
    return (1. / den) if (den > 0) else 0

def l_bfgs(X, A, f, m=8):
    old_grad = None

    y = list()
    s = list()
    alpha = list()

    k = -1
    while True:
        pred = f(A, X)
        grad = 2 * A * pred

        if k >= m:
            q = np.copy(grad)
            for s_i, y_i, alpha_i in reversed(list(zip(s, y, alpha))):
                q -= alpha_i * y_i

            den = np.dot(y[-1], y[-1])
            if den > 0:
                z = inner_product(y[-1] / den, s[-1], q)
            else:
                z = 0

            for s_i, y_i, alpha_i in zip(s, y, alpha):
                rho_i = compute_rho(s_i, y_i)
                beta_i = rho_i * np.dot(y_i, z)
                z += s_i * (alpha_i - beta_i)
        else:
            z = grad

        # Line search
        c1 = 1e-04
        steplength = 1.0
        while f(A - steplength*z, X) > pred - c1*steplength*np.dot(grad, z):
            steplength /= 2

        # Update weights
        delta = steplength * z
        A -= delta
        yield pred

        if old_grad is not None:
            y.append(grad - old_grad)
            s.append(delta)
            rho_i = compute_rho(s[-1], y[-1])
            alpha.append(rho_i * np.dot(s[-1], grad))

            if len(y) > m:
                y = y[1:]
                s = s[1:]
                alpha = alpha[1:]

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