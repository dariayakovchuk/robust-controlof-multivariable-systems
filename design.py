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
    num = [0, 2.]
    den = [1, -.9]
    # sys1 = tf(num, den, Ts)  # SISO system
    sys1 = tf([[[n * 2 for n in num], [n * 0.1 for n in num]], [[n * 0 for n in num], num]],
              [[den, den], [den, den]], Ts)  # MIMO system
    return ss(sys1)


def synth_h2(g, ejw, w, w1, w2):
    n = 3
    x, y, gamma = cp.Variable((n, 1)), cp.Variable((n, 1)), cp.Variable((len(w), 1), nonneg=True)
    GAMMA_prev = 0.
    cost = 0
    constraints = []
    for i in range(len(w)):
        G = g[i]
        # X = x[0] * (ejw[i]**2) + x[1] * ejw[i] + x[2]
        # Y = 1 * (ejw[i]**2) + y[1] * ejw[i] + y[2]
        X = cp.bmat([[x[0] * (ejw[i] ** 2) + x[1] * ejw[i] + x[2], 0], [0, x[0] * (ejw[i] ** 2) + x[1] * ejw[i] + x[2]]])
        Y = cp.bmat([[1 * (ejw[i] ** 2) + y[1] * ejw[i] + y[2], 0], [0, 1 * (ejw[i] ** 2) + y[1] * ejw[i] + y[2]]])
        GAMMA = gamma[i]
        Xc, Yc = 0.1 * (ejw[i]**2), ejw[i]**2
        # P, Pc = Y + G * X, Yc + G * Xc
        I2 = np.eye(2)
        P, Pc = Y + G @ X, (I2*Yc) + G @ (I2*Xc)
        T = P.H @ Pc + Pc.conjugate().transpose() @ P - Pc.conjugate().transpose() @ Pc
        f = cp.vstack([w1*Y, w2*X])
        I = np.eye(4)
        # tmp = cp.vstack([cp.hstack([GAMMA, w1*Y]), cp.hstack([conj(w1*Y), T])])
        # tmp = cp.vstack([cp.hstack([(I*GAMMA)[0], f[0]]), cp.hstack([(I*GAMMA)[1], f[1]]), cp.hstack([conj(f), T])])
        tmp = cp.vstack([cp.hstack([(I*GAMMA), f]), cp.hstack([f.H, T])])
        if i == 0:
            cost += (GAMMA * w[0]) / 2
        else:
            cost += ((GAMMA_prev + GAMMA) / 2) * (w[i] - w[i-1])
        constraints += [tmp >> 0]
        GAMMA_prev = GAMMA
    obj = cp.Minimize(2 * cost)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)
    print("status:", prob.status)
    print("optimal value", prob.value)
    return x.value, y.value

def freq_response(g, w):
    mag, phase, omega = freqresp(g, w)
    # print('mag, phase, omega', mag, phase, omega)
    ejw = np.exp(1j * w * Ts)
    # sjw = mag * np.exp(1j * phase)
    sjw = (mag * np.exp(1j * phase)).transpose(2, 0, 1)
    return sjw, ejw

def H2_perf(t, w1, w2):
    g = plant()
    G, ejw = freq_response(g, t)
    K = synth_h2(G, ejw, t, w1, w2)
    print(K)

def design():
    """Show results of designs"""
    W1 = 1.
    W2 = 1.
    t = np.logspace(np.log10(0.01), np.log10(math.pi/Ts), 10) # (a,b) 10^a ..... 10 ^b   -> c ... d ---> a = log(c), b = log(d)
    H2_perf(t, W1, W2)

if __name__ == "__main__":
    # analysis()
    design()