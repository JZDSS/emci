from utils.nme import NME
from utils.auc import AUC


def metric(add):
    def r(self, *args, **kwargs):
        add(self, *args, **kwargs)
        return self
    return r


class Metrics(object):
    def __int__(self):
        self.nme = None
        self.auc = None

    @metric
    def add_nme(self, decay=0.999):
        self.nme = NME(decay)

    @metric
    def add_auc(self, low=0, high=0.08, step=0.01):
        self.auc = AUC(low, high, step)
