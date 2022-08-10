from control.matlab import *


class MIOSystem():
    def __init__(self, Ts, g_data):
        self.Ts = Ts
        self.G = None
        self.create_plant(g_data)

    def create_plant(self, g):
        return self.G

    def freq_response(self): pass