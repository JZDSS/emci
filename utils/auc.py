import numpy as np


class AUC(object):
    def __init__(self, low=0, high=0.08, step=0.01, decay=0.9):
        if low < 0:
            low = 0
        if high > 1:
            high = 1
        self.low = low
        self.high = high
        self.step = step
        self.decay = decay
        self.value = None

    def update(self, nme):
        """
            计算AUC
            :param nme:normalised mean error ,形状为M维数组,M为图片数量
            :return: AUC:area-under-the-curve ,标量
            """
        n = nme.size

        count = np.zeros(int((self.high - self.low) / self.step) + 1)
        bound = np.linspace(self.low, self.high, len(count))
        for i in range(n):
            v = nme[i]
            for j in range(len(bound)):
                if v <= bound[j]:
                    count[j] += 1
                    break

        q = np.cumsum(count)
        area = q / n

        area = (area[0:-1] + area[1:]) * self.step / 2

        auc = np.sum(area) / (self.high - self.low)

        if self.value is None:
            self.value = auc
        else:
            # 指数滑动窗口
            self.value *= self.decay
            self.value += (1 - self.decay) * auc

        return auc

    def clear(self):
        self.value = None

if __name__ == '__main__':
    a = AUC()
    a.update(0.01*np.ones(100))