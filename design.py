import math
import os
from mosek.fusion import *

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
    sys1 = tf([[[n * 2 for n in num], [n * 0.1 for n in num]], [[n * 0 for n in num], num]],
              [[den, den], [den, den]], Ts)  # MIMO system
    return ss(sys1)


def synth_h2(g, ejw, w, w1, w2):
    M = Model("semidefinite model")
    n = 3
    x, y, gamma = M.variable([n, 1]), M.variable([n, 1]), M.variable([len(w), 1], Domain.greaterThan(0.))
    GAMMA_prev = 0.
    cost = 0
    constraints = []
    for i in range(len(w)):
        G = g[i]
        X = Expr.outer(x.index([0, 0]), ejw[i] ** 2)
        Y = [[Expr.mul(1 * (ejw[i] ** 2)) + y[1] * ejw[i] + y[2], 0], [0, 1 * (ejw[i] ** 2) + y[1] * ejw[i] + y[2]]]
        GAMMA = gamma[i]
        Xc, Yc = 0.1 * (ejw[i]**2), ejw[i]**2
        I2 = np.eye(2)
    #     P, Pc = Y + G @ X, (I2*Yc) + G @ (I2*Xc)
    #     T = P.H @ Pc + Pc.conjugate().transpose() @ P - Pc.conjugate().transpose() @ Pc
    #     f = cp.vstack([w1*Y, w2*X])
        I = np.eye(4)
    #     tmp = cp.vstack([cp.hstack([(I*GAMMA), f]), cp.hstack([f.H, T])])
        if i == 0:
            cost += (GAMMA * w[0]) / 2
        else:
            cost += ((GAMMA_prev + GAMMA) / 2) * (w[i] - w[i-1])
        M.constraint((tmp, Domain.inPSDCone()))
        GAMMA_prev = GAMMA
    M.objective(ObjectiveSense.Minimize, 2 * cost)    # obj = cp.Minimize(2 * cost)
    M.solve()
    # return x.value, y.value

def freq_response(g, w):
    mag, phase, omega = freqresp(g, w)
    ejw = np.exp(1j * w * Ts)
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
    design()