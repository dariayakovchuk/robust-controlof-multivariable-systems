import numbers

import numpy as np
import cvxpy as cp
import re


class Controller:
    def __init__(self, G, Ts, w, weights_input, n, Kc):
        self.x = None
        self.y = None
        self.K = None
        self.ejw = None
        self.weights = None
        self.var = None
        self.weights_shape = None
        self.valid = False
        self.shape = 2
        self.Gjw = G
        self.Ts = Ts
        self.w = w
        self.Kc = Kc
        self.n = n
        self.structure_input(weights_input)
        self.define_complex_exp_function()

    def structure_input(self, obj):
        weights = []
        var = []
        for i in range(len(obj)):
            weights.append(float((re.findall(r"[-+]?(?:\d*\.\d+|\d+)", obj[i]))[0]))
            var.append(''.join(x for x in obj[i] if x =='x' or x =='y'))
        if len(weights) == len(var):
            self.define_weights(weights)
            self.var = var
        else:
            print('Wrong input')

    def XY(self, XY_input, x, y):      # TO DO
        X = cp.bmat([[x, 0], [0, x]])
        Y = cp.bmat([[y, 0], [0, y]])
        return X, Y

    def define_weights(self, weights):
        self.weights = np.array(weights)
        self.weights_shape = self.weights.shape[0]

    def eye_matrix_shape(self):
        self.validation()
        if self.valid:
            iG = self.Gjw[0].shape[0]
            self.shape = iG

    def discrete_time_controller_X(self, x, t):
        tmp = 0.
        for i in range(self.n):
            tmp += x[i] * (self.ejw[t] ** i)
        return tmp

    def discrete_time_controller_Y(self, y, t):
        tmp = 0.
        for i in range(self.n):
            if i == self.n - 1:
                tmp += 1 * (self.ejw[t] ** i)
            else:
                tmp += y[i] * (self.ejw[t]**i)
        return tmp

    def define_complex_exp_function(self):
        self.ejw = np.exp(1j * self.w * self.Ts)

    def F(self, X, Y):
        array = []
        for i in range(len(self.weights)):
            if self.var[i] == 'x':
                array.append(self.weights[i] * X)
            elif self.var[i] == 'y':
                array.append(self.weights[i] * Y)
        return cp.vstack(array)

    def validation(self):
        iG, jG = self.Gjw[0].shape[0], self.Gjw[0].shape[1]
        if isinstance(self.K, numbers.Number):
            self.valid = True
        elif isinstance(self.K[0], numbers.Number):
            self.K = self.K[0]
            self.valid = True
        else:
            iK, jK = self.K.shape[0], self.K.shape[1]
            if jG == iK or jK == 0:
                self.valid = True

    def optimization(self, structureXY):
        x, y, gamma = cp.Variable((self.n, 1)), cp.Variable((self.n, 1)), cp.Variable((len(self.w), 1), nonneg=True)

        GAMMA_prev, cost = 0, 0
        constraints = []
        for i in range(len(self.w)):
            X1 = self.discrete_time_controller_X(x, i)
            Y1 = self.discrete_time_controller_Y(y, i)
            X, Y = self.XY(structureXY, X1, Y1)

            GAMMA = gamma[i]
            Xc, Yc = self.Kc * (self.ejw[i] ** 2), self.ejw[i] ** 2

            self.eye_matrix_shape()
            I = np.eye(self.shape)
            I2 = np.eye(2 * self.shape)
            f = self.F(X, Y)
            P = Y + self.Gjw[i] @ X
            Pc = (I * Yc) + self.Gjw[i] @ (I * Xc)
            T = P.H @ Pc + Pc.conjugate().transpose() @ P - Pc.conjugate().transpose() @ Pc
            tmp = cp.vstack([cp.hstack([(I2 * GAMMA), f]), cp.hstack([f.H, T])])
            if i == 0:
                cost += (GAMMA * self.w[0]) / 2
            else:
                cost += ((GAMMA_prev + GAMMA) / 2) * (self.w[i] - self.w[i - 1])
            constraints += [tmp >> 0]
            GAMMA_prev = GAMMA
        obj = cp.Minimize(2 * cost)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK, verbose=True)
        print("status:", prob.status)
        print("optimal value", prob.value)
        print(x.value, y.value)
        K = x.value.T @ np.linalg.pinv(y.value).T
        return K[0], prob.value, prob.status

    @staticmethod
    def check_accord(val1, val2, val3, err):
        condition1 = (val1 < err)
        condition2 = (val2 < err)
        condition3 = (val3 == 'infeasible' or val3 == 'unbounded')
        if condition1 or condition2 or condition3:
            return True
        else:
            return False

    def iterative_solve(self, XY_structure):
        e = 0.001
        gamma_prev = 0
        self.K = self.Kc
        self.validation()
        while self.valid:
            self.K, gamma, status = self.optimization(XY_structure)
            if self.check_accord(gamma, gamma - gamma_prev, status, e):
                break
            gamma_prev = gamma
            self.Kc = self.K