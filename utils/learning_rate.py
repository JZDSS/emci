class exponential_decay(object):

    def __init__(self, learning_rate, decay_steps, decay_rate, global_step=1, staircase=False, name=None):
        self.learning_rate = learning_rate
        self.global_step = global_step
        if staircase:
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
        else:
            self.decay_steps = 1
            self.decay_rate = decay_rate ** (1 / decay_steps)

    def get(self):
        if self.global_step % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate
        self.global_step += 1
        return self.learning_rate


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
        global_step = min(self.global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) *
        (1 - global_step / self.decay_steps) ** (self.power) + self.end_learning_rate
        self.global_step += 1
        if self.cycle:
            self.global_step = self.global_step % self.decay_steps + 1
        return decayed_learning_rate

# class lr(object):
#     def __init__(self, config):
#         """
#         :param config: A dictionary.
#         config = {0: polynomial_decay(1e-8, 2000, 0.01),
#                   2001: exponential_decay(0.01, 300, 0.99)}
#         """
#         self.config = config
#
#
#     def get(self):
#         pass