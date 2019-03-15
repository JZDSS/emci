import torch
import torch.nn as nn
from layers.module.boundary_loss import BoundaryLossN

class LocalLoss(nn.Module):

    def __init__(self):
        super(LocalLoss, self).__init__()
        self.criterion_local = BoundaryLossN(version='soft', alpha=10, threshold=0.05, threshold_decay=0.99)
        self.criterion_global = BoundaryLossN(version='soft', alpha=10, threshold=0.05, threshold_decay=0.99)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, out_local, prob_local, out_global, prob_global, landmarks, mask):
        # local
        active_mask = mask > 0
        unactive_mask = mask <= 0

        # active
        coord = out_local[active_mask.expand_as(out_local)].view(-1, 212)
        prob = prob_local[active_mask.expand_as(prob_local)].view(-1, 212)

        loss_coo_local = self.criterion_local(coord, landmarks)
        loss_prob_local_act = self.mse(prob,
                                       torch.exp(- 14 * (coord - landmarks) ** 2)).sum(dim=1).mean()

        # unactive
        prob = prob_local[unactive_mask.expand_as(prob_local)].view(-1, 212 * 48)
        loss_prob_local_unact = 0.01 * (prob ** 2).sum(dim=1).mean()

        # global
        loss_coo_global = self.criterion_global(out_global, landmarks)
        loss_prob_global = self.mse(prob_global,
                                    torch.exp(- 14 * (out_global - landmarks) ** 2)).sum(dim=1).mean()

        return loss_coo_local, loss_prob_local_act, loss_prob_local_unact, \
               loss_coo_global, loss_prob_global

if __name__ == '__main__':
    from data.bbox_dataset import BBoxDataset
    from torch.utils.data import DataLoader
    a = BBoxDataset('/data/icme/crop/data/picture',
                    '/data/icme/crop/data/landmark',
                    '/data/icme/train', phase='train',
                    max_jitter=30, max_angle=30)
    batch_iterator = iter(DataLoader(a, batch_size=8, shuffle=True, num_workers=0))
    images, landmarks, mask = next(batch_iterator)
    pred_coord = torch.randn(8, 106, 2, 7, 7)
    pred_prob = torch.randn(8, 106, 1, 7, 7)
    # target = torch.randn(8, 106, 2, 7, 7)
    crit = LocalLoss()
    crit(pred_coord, landmarks, mask)
