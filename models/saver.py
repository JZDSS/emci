import os
import torch


class Saver:
    def __init__(self, dir, model_name, max_keep):
        self.dir = dir
        self.model_name = model_name
        self.max_keep = max_keep
        self.cache = []

    def save(self, state, i):
        path = os.path.join(self.dir, '%s-%d.pth' % (self.model_name, i))
        torch.save(state, path)
        self.cache.append(path)
        if len(self.cache) > self.max_keep:
            os.remove(self.cache[0])
            del self.cache[0]
