import os
import torch
import logging
import glob


class Saver_:
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
        names = glob.glob(os.path.join(self.dir, '%s-*.pth' % self.model_name))
        if not names:
            return None
        if 'tmp' in names[-1]:
            del names[-1]
        if not names:
            return None
        names = [os.path.basename(n) for n in names]
        idx = [int(name.split('.')[0].split('-')[-1]) for name in names]
        max_idx = max(idx)
        return '%s-%d.pth' % (self.model_name, max_idx)

    def load(self, model, file):
        state_dict = torch.load(os.path.join(self.dir, file))
        model.load_state_dict(state_dict)

    def load_last_ckpt(self, model):
        name = self.last_ckpt()
        if name is None:
            return
        path = os.path.join(self.dir, name)
        if path is None:
            logging.warning("No existed checkpoint found!")
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)


class Saver:
    def __init__(self, dir, keys, max_keep=None):
        self.keys = keys
        self.s = {}
        for k in keys:
            self.s[k] = Saver_(dir, k, max_keep)

    def save(self, state, i):
        for k in self.keys:
            self.s[k].save(state[k], i)

    def load(self, model_dic, iter):
        for k in model_dic.keys():
            file = self.s[k].model_name + '-%d.pth' % iter
            self.s[k].load(model_dic[k], file)

    def load_last_ckpt(self, model_dic):
        for k, v in model_dic.items():
            self.s[k].load_last_ckpt(v)


if __name__ == '__main__':
    from models.dense201 import Dense201
    s = Saver('../exp/soft10boundaryN-align-j3-test/ckpt', ['model', 'opt'])
    net = Dense201()
    opt = torch.optim.Adam(net.parameters())
    # s.load({'model': net,
    #         'opt': opt}, 3800)
    s.load_last_ckpt({'model': net,
                      'opt': opt})

    print(s.last_iter())
