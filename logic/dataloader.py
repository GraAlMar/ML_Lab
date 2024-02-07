import numpy as np


class DataLoader:
    def __init__(self, target):
        self._Target = target
        self.X, self.Y = np.loadtxt(target, skiprows=1, unpack=True)

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y
