import math
from proto import all_pb2

class exponential_decay(object):

    def __init__(self, learning_rate, decay_steps, decay_rate, global_step=1, staircase=False, name=None):
        self.learning_rate_ = learning_rate
        self.global_step = global_step

        if staircase:
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
        else:
            self.decay_steps = 1
            self.decay_rate = decay_rate ** (1 / decay_steps)

        self.learning_rate = self.learning_rate_ * self.decay_rate** (self.global_step // self.decay_steps)

    def get(self):
        if self.global_step % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate
        self.global_step += 1
        return self.learning_rate

    def set_global_step(self, global_step):
        self.global_step = global_step
        self.learning_rate = self.learning_rate_ * self.decay_rate ** (self.global_step // self.decay_steps)

class piecewise_constant(object):

    def __init__(self, boundaries, values, global_step=1):
        assert len(values) == len(boundaries) + 1
        self.global_step = global_step
        self.values = values
        self.boundaries = boundaries
        self.learning_rate = values[0]
        for i, b in enumerate(boundaries):
            if self.global_step >= b:
                self.learning_rate = values[i + 1]

    def get(self):
        if self.global_step in self.boundaries:
            self.learning_rate = self.values[self.boundaries.index(self.global_step) + 1]
        self.global_step += 1
        return self.learning_rate

    def set_global_step(self, global_step):
        self.global_step = global_step
        self.learning_rate = self.values[0]
        for i, b in enumerate(self.boundaries):
            if self.global_step >= b:
                self.learning_rate = self.values[i + 1]

class polynomial_decay(object):
    def __init__(self, learning_rate, decay_steps,
                 end_learning_rate=0.0001, power=1.0,
                 global_step=1, cycle=False):
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate=end_learning_rate
        self.power = power
        self.global_step = global_step
        self.cycle = cycle

    def get(self):
        if self.cycle:
            decay_steps = self.decay_steps * math.ceil(self.global_step / self.decay_steps)
            global_step = self.global_step
        else:
            decay_steps = self.decay_steps
            global_step = min(self.global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) * \
            (1 - global_step / decay_steps) ** (self.power) + self.end_learning_rate
        self.global_step += 1
        return decayed_learning_rate

    def set_global_step(self, global_step):
        self.global_step = global_step

class mix(object):

    def __init__(self, config, global_step=1):
        """
        :param config: A dictionary.
        for example:
        config = ={0: polynomial_decay(1e-8, 5000, 0.01),
                   5001: exponential_decay(0.01, 300, 0.9)}
        means learning rate increase linearly in first 5000 steps and decay exponentially later.
        """
        self.boundaries, self.values = zip(*config.items())
        self.global_step = global_step

        self.curr = self.values[0]
        self.curr.set_global_step(global_step)
        for i, b in enumerate(self.boundaries):
            if self.global_step >= b:
                global_step -= b
                self.curr = self.values[i]
                self.curr.set_global_step(global_step)

    def get(self):
        self.global_step += 1
        if self.global_step in self.boundaries:
            self.curr = self.values[self.boundaries.index(self.global_step)]
        return self.curr.get()

    def set_global_step(self, global_step):
        self.global_step = global_step
        self.curr = self.values[0]
        self.curr.set_global_step(global_step)
        for i, b in enumerate(self.boundaries):
            if self.global_step >= b:
                global_step -= b
                self.curr = self.values[i]
                self.curr.set_global_step(global_step)
def get_exp(cfg):
    return exponential_decay(learning_rate=cfg.start_lr,
                                           decay_steps=cfg.decay_steps,
                                           decay_rate=cfg.decay_rate,
                                           staircase=cfg.staircase)


def get_pie(cfg):
    boundaries = [b for b in cfg.boundary]
    values = [v for v in cfg.value]
    return piecewise_constant(boundaries, values)


def get_pol(cfg):
    return polynomial_decay(learning_rate=cfg.learning_rate,
                                          decay_steps=cfg.decay_steps,
                                          end_learning_rate=cfg.end_learning_rate,
                                          power=cfg.power,
                                          cycle=cfg.cycle)


def get_mix(cfg):
    mix_cfg = {}
    for item in cfg.item:
        mix_cfg[item.boundary] = get_lr(item.lr)
    return mix(mix_cfg)


def get_lr(cfg):

    if cfg.type == all_pb2.LearningRate.EXPONENTIAL:
        return get_exp(cfg.exponential_decay)
    elif cfg.type == all_pb2.LearningRate.PIECEWISE:
        return get_pie(cfg.piecewise_constant)
    elif cfg.type == all_pb2.LearningRate.POLYNOMIAL:
        return get_pol(cfg.polynomial_decay)
    elif cfg.type == all_pb2.LearningRate.MIX:
        return get_mix(cfg.mix)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config = {0: polynomial_decay(1e-8, 5000, 0.01),
              5001: exponential_decay(0.01, 300, 0.9)}
    l = mix(config, global_step=3500)
    # l = polynomial_decay(1e-2, 5000, 1e-6, power=0.5, cycle=True)
    lrs = []
    for i in range(10000):
        lrs.append(l.get())
    x = list(range(10000))
    plt.plot(x, lrs)
    plt.show()

