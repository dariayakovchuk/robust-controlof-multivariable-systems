import math
import os

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from control.matlab import *
from control import freqresp
from cvxpy import conj, real

Ts = 0.1


def plant():
    """
    plant() -> G - LTI object
    """
    num = [[[0, 2.]]]
    den = [[[1, -.9]]]
    sys1 = tf(num, den)
    return ss(sys1)


def synth_h2(G, w, w_1, w1, w2):
    n = 3
    x, y, z = cp.Variable(n, complex=True), cp.Variable(n, complex=True), cp.Variable(n, complex=True)
    X = x[0] * (w**2) + x[1] * w + x[2]
    Y = 1 * (w**2) + y[1] * w + y[2]
    Z = z[0] * (w**2) + z[1] * w + z[2]
    Z_w_1 = z[0] * (w_1 ** 2) + z[1] * w_1 + z[2]
    Xc, Yc = 0.1 * (w**2), (w**2)
    P, Pc = Y + G*X, Yc + G * Xc
    T = conj(P) * Pc + conj(Pc) * P - conj(Pc) * Pc
    # F = cp.hstack([w1*Y, w2*X])
    F = w1*Y
    I = np.eye(2)
    tmp = cp.vstack([cp.hstack([Z, F]), cp.hstack([conj(F), T])])
    constraints = [tmp >> 0]
    obj = cp.Minimize((real(Z)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", X.value, Y.value)
    K = Y.value / X.value
    return K


def freq_response(g, w):
    mag, phase, omega = freqresp(g, w)
    sjw = (mag * np.exp(1j * phase * Ts))
    return omega, sjw


def H2_perf(t, w1, w2):
    g = plant()
    G, w = freq_response(g, t)
    for i in range(len(w)):
        K = synth_h2(G[i], w[i], w[i-1], w1, w2)



def design():
    """Show results of designs"""
    W1 = 1.
    W2 = 1.
    t = np.logspace(0.01, math.pi/Ts, 400)
    H2_perf(t, W1, W2)
    H2_perf(t, W1, W2)
    H2_perf(t, W1, W2)


if __name__ == "__main__":
    # analysis()
    design()