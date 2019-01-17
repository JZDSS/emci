import argparse
from data.face_dataset import FaceDataset
from layers.module.wing_loss import WingLoss
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn


parser = argparse.ArgumentParser(
    description='Landmark Detection Training')

parser.add_argument('-l', '--learning_rate', default=1e-3)
parser.add_argument('-b', '--batch_size', default=16)
parser.add_argument('-c', '--cuda', default=True)
parser.add_argument('-n', '--n_gpu', default=1)

args = parser.parse_args()

if __name__ == '__main__':
    print(args.cuda)
    net =models.resnet18(num_classes=212)
    sig = nn.Sigmoid()
    a = FaceDataset("/data/icme", "/data/icme/train")
    batch_iterator = iter(DataLoader(a, batch_size=4, shuffle=True, num_workers=4))

    criterion = WingLoss(10, 0.5)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    running_loss = 0.0
    batch_size = 4
    epoch_size = len(a) // batch_size
    for iteration in range(100000):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(a, batch_size,
                                                  shuffle=True, num_workers=4))
        # load train data
        images, landmarks = next(batch_iterator)
        # images = images.cuda()
        # landmarks = landmarks.cuda()
        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        # if args.cuda:
        #     images = Variable(images.cuda())
        #     landmarks = [Variable(anno.cuda()) for anno in landmarks]
        # else:
        #     images = Variable(images)
        #     landmarks = [Variable(anno) for anno in landmarks]
        # forward
        out = net(images)
        out = sig(out)
        # backprop
        optimizer.zero_grad()
        loss = criterion(out, landmarks)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print(loss.item())