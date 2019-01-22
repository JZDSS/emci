from models import dense201
import torch.nn as nn
from models import saver
from utils import alignment

class IterModel(nn.Module):

    def __init__(self):
        super(IterModel, self).__init__()
        self.model = dense201.Dense201()
        saver.Saver('backup', 'model').load(self.model, 'aligned_densenet-166400.pth')
        self.aligner = alignment.Align('cache/mean_landmarks.pkl', scale=(224, 224), margin=(0.15, 0.1))

    def forward(self, x):
        landmark = self.model(x)
        aligned = self.aligner(x.cpu().data.numpy(), landmark.cpu().data.numpy(), [0, 0, 223, 223]).cuda()
        landmark = self.model(aligned)
        return landmark
