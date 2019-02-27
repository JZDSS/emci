import os
import torch
import logging

class Saver:
    def __init__(self, dir, model_name='model', max_keep=None):
        """
        :param dir: 模型保存路径
        :param model_name: 模型名
        :param max_keep: 保存最近的max_keep个模型
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir
        self.model_name = model_name
        self.max_keep = max_keep
        self.cache = []

    def save(self, state, i):
        path = os.path.join(self.dir, '%s-%d.pth' % (self.model_name, i))
        torch.save(state, path + '.tmp')
        if not self.max_keep is None:
            self.cache.append(path)
            if len(self.cache) > self.max_keep:
                os.remove(self.cache[0])
                del self.cache[0]
        os.rename(path + '.tmp', path)

    def last_ckpt(self):
        names = os.listdir(self.dir)
        if not names:
            return None
        if 'tmp' in names[-1]:
            del names[-1]
        if not names:
            return None
        idx = [int(name.split('.')[0].split('-')[-1]) for name in names]
        max_idx = max(idx)
        return '%s-%d.pth' % (self.model_name, max_idx)

    def load(self, model, file):
        state_dict = torch.load(os.path.join(self.dir, file))
        model.load_state_dict(state_dict)

    def load_last_ckpt(self, model):
        path = os.path.join(self.dir, self.last_ckpt())
        if path is None:
            logging.warning("No existed checkpoint found!")
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
