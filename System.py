from control.matlab import *
import numpy as np


class MIOSystem:
    def __init__(self, Ts, g_data, t):
        self.Ts = Ts
        self.t = t
        self.G = None
        self.G_shape = None
        self.param = None
        self.create_plant(g_data)

    def check_input(self):
        self.G_shape = [2, 2]

    def create_plant(self, g):
        return self.G

    def select_param(self, m):
        a, b, c = 0, 0, 0
        for i in range(G.shape)
        m_shape0, m_shape1, m_shape2 = m.shape[0], m.shape[1], m.shape[2]

        return a, b, c

    def freq_response(self):
        mag, phase, omega = freqresp(self.G, self.t)
        a, b, c = self.select_param(mag, phase, omega)
        sjw = (mag * np.exp(1j * phase)).transpose(2, 0, 1)
        return sjw
