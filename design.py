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


def synth_h2(g, t, w1, w2):
    n = 3
    x, y, z = cp.Variable(n), cp.Variable(n), cp.Variable(n)
    Z_prev = 0.
    cost = 0
    constraints, K = [], []
    for i in range(len(t)):
        G = g[i]
        w = t[i]
        X = x[0] * (w**2) + x[1] * w + x[2]
        Y = 1 * (w**2) + y[1] * w + y[2]
        Z = z[0] * (w**2) + z[1] * w + z[2]
        Xc, Yc = 0.1 * (w**2), (w**2)
        P, Pc = Y + G * X, Yc + G * Xc
        T = conj(P) * Pc + conj(Pc) * P - conj(Pc) * Pc
        # f = cp.hstack([w1*Y, w2*X])
        F = w1*Y
        I = np.eye(2)
        tmp = cp.vstack([cp.hstack([Z, F]), cp.hstack([conj(F), T])])
        # tmp2 = cp.vstack([cp.hstack([I*Z, f]), cp.hstack([conj(f), T])])
        cost += (Z_prev + Z) * (t[i]-t[i-1]) / 2
        constraints += [tmp >> 0]
        Z_prev = Z
    obj = cp.Minimize(2 * real(cost))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    return K

def freq_response(g, w):
    mag, phase, omega = freqresp(g, w)
    # print('mag, phase, omega', mag, phase, omega)
    ejw = np.exp(1j * w * Ts)
    sjw = mag * np.exp(1j * phase)
    return sjw, ejw

def H2_perf(t, w1, w2):
    g = plant()
    G, w = freq_response(g, t)
    K = synth_h2(G, w, w1, w2)
    print(K)

def design():
    """Show results of designs"""
    W1 = 1.
    W2 = 1.
    t = np.logspace(0.01, math.pi/Ts, 2)
    H2_perf(t, W1, W2)

if __name__ == "__main__":
    # analysis()
    design()