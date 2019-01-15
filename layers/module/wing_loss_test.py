import torch
import numpy as np
import matplotlib.pyplot as plt

from layers.module.wing_loss import WingLoss


if __name__ == '__main__':

    # forward test
    N = 1000
    x = np.ones(shape=(N))
    y = 20 * np.random.randn(N) + x
    criterion = WingLoss(10, 0.5)

    loss = criterion(torch.tensor(x), torch.tensor(y))

    plt.scatter(y - x, loss.numpy())
    plt.show()

    # backward test
    x = torch.tensor(x, requires_grad=False).cuda()
    y = torch.tensor(y, requires_grad=False).cuda()

    delta = torch.ones_like(x, requires_grad=True)

    opt = torch.optim.Adam([delta], 0.01)

    for i in range(10000):
        opt.zero_grad()
        loss = criterion(x + delta, y)
        loss.backward()
        opt.step()

    learned = delta.cpu().data.numpy()
    ground_truth = (y - x).cpu().data.numpy()

    print(learned - ground_truth)
