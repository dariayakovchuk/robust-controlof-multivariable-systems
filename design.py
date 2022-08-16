import math
import os
from mosek.fusion import *
from mosek import LinAlg

import numpy as np
import matplotlib.pyplot as plt

from control.matlab import *
from control import freqresp

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


def synth_h2(g, eiw, w, w1, w2):
    M = Model("semidefinite model")
    n = 3
    x, y, gamma = M.variable([n, 1]), M.variable([n, 1]), M.variable([len(w), 1], Domain.greaterThan(0.))
    GAMMA_prev = 0.
    cost = 0
    for i in range(len(w)):
        G = [g[i].real, g[i].imag]
        ejw = [eiw[i].real, eiw[i].imag]
        X1 = [Expr.add([Expr.mul(x.index(0, 0), ejw[0] ** 2), Expr.mul(x.index(1, 0), ejw[0]), x.index(2, 0)]),
              Expr.add([Expr.mul(x.index(0, 0), ejw[1] ** 2), Expr.mul(x.index(1, 0), ejw[1]), x.index(2, 0)])]
        Y1 = [Expr.add([Expr.mul(y.index(0, 0), ejw[0] ** 2), Expr.mul(y.index(1, 0), ejw[0]), y.index(2, 0)]),
              Expr.add([Expr.mul(y.index(0, 0), ejw[1] ** 2), Expr.mul(y.index(1, 0), ejw[1]), y.index(2, 0)])]
        X_REAL, X_IMAG = Expr.vstack(Expr.hstack(X1[0], 0), Expr.hstack(0, X1[0])), Expr.vstack(Expr.hstack(X1[1], 0),Expr.hstack(0, X1[1]))
        Y_REAL, Y_IMAG = Expr.vstack(Expr.hstack(Y1[0], 0), Expr.hstack(0, Y1[0])), Expr.vstack(Expr.hstack(Y1[1], 0), Expr.hstack(0, Y1[1]))
        GAMMA = gamma.index(i, 0)
        Xc, Yc = [0.1 * ejw[0] ** 2, 0.1 * ejw[1] ** 2], [ejw[0] ** 2, ejw[1] ** 2]
        I2 = np.eye(2)
        P, Pc = [Expr.add(Y_REAL, Expr.mul(G[0], X_REAL)), Expr.add(Y_IMAG, Expr.mul(G[1], X_REAL))], [Yc[0] * I2 + G[0] * I2, Yc[1] * I2 + G[1] * I2]
        T_REAL = Expr.sub(Expr.add(Expr.mul(P[0], Pc[0]), Expr.mul(Pc[0], P[0])), Pc[0] * Pc[0])
        T_IMAG = Expr.sub(Expr.add(Expr.mul(Expr.neg(P[1]), Pc[1]), Expr.mul(-Pc[1], P[1])), -Pc[1] * Pc[1])
        f_real = Expr.vstack(Expr.mul(w1, Y_REAL), Expr.mul(w2, X_REAL))
        f_imag = Expr.vstack(Expr.mul(w1, Y_IMAG), Expr.mul(w2, X_IMAG))
        I = Matrix.eye(4)
        expr1 = Expr.mul(I, GAMMA)
        tmp_real = Expr.vstack(Expr.hstack(expr1, f_real), Expr.hstack(Expr.transpose(f_real), T_REAL))
        tmp_imag = Expr.vstack(Expr.hstack(expr1, f_imag), Expr.hstack(Expr.transpose(f_imag), T_IMAG))
        if i == 0:
            cost = Expr.mul(Expr.mul(GAMMA, w[0]), 1 / 2)
        else:
            cost = Expr.add(Expr.mul(Expr.mul(Expr.add(GAMMA_prev, GAMMA), w[i] - w[i - 1]), 1 / 2), cost)
        M.constraint('{0}'.format(i), Expr.vstack(Expr.hstack(tmp_real, Expr.neg(tmp_imag)), Expr.hstack(tmp_imag, tmp_real)), Domain.inPSDCone(12))
        GAMMA_prev = GAMMA
    M.objective(ObjectiveSense.Minimize, Expr.mul(2, cost))
    M.solve()
    print(x.level())
    print(y.level())
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
    t = np.logspace(np.log10(0.01), np.log10(math.pi / Ts),
                    100)  # (a,b) 10^a ..... 10 ^b   -> c ... d ---> a = log(c), b = log(d)
    H2_perf(t, W1, W2)


if __name__ == "__main__":
    design()
