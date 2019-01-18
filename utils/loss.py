import numpy as np


class Loss(object):
    def __init__(self, decay=0.9):
        self.decay = decay
        self.value = None

    def update(self, loss):
        if self.value is None:
            self.value = loss
        else:
            # 指数滑动窗口
            self.value *= self.decay
            self.value += (1 - self.decay) * loss

    def clear(self):
        self.value = None
