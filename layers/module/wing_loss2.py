from layers.module import wing_loss
from torch import nn
class WingLoss2(nn.Module):
    def __init__(self, w1, epsilon1, w2, epsilon2):
        super(WingLoss2, self).__init__()
        self.wing1 = wing_loss.WingLoss(w1, epsilon1)
        self.wing2 = wing_loss.WingLoss(w2, epsilon2)

    def forward(self, predictions, targets):
        loss1 = self.wing1(predictions[:34, :], targets[:34, :])
        loss2 = self.wing2(predictions[34:, :], targets[34:, :])
        return loss1 + loss2
