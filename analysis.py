import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from design import plant, Ts
from control import step_response
from control.matlab import *
K = 0.1


def analysis():
    """Plot open-loop responses for various inputs"""
    g = plant()
    G = 1 + g*K
    t = np.logspace(0.01, math.pi/Ts, 400)

    # step response of the system
    plt.figure(1)
    T, yout = step_response(G, input=0)
    plt.plot(T.T, yout.T)
    plt.show(block=False)

    # Bode plot for the system
    plt.figure(2)
    bode(G)
    print('margins', margin(G))
    # mag, phase, om = bode(G)
    plt.show(block=False)

    # Nyquist plot for the system
    plt.figure(3)
    # nyquist_plot(G, t)
    nyquist(G)
    print(G.poles(), G.zeros())
    plt.show(block=False)

    # Root lcous plot for the system
    plt.figure(4)
    rlocus(G, t)
    for pole in G.pole():
        plt.plot(np.real(pole), np.imag(pole), 'rs')
    plt.grid()
    plt.show(block=False)

analysis()