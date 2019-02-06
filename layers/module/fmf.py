import torch.nn as nn
import torch
import torch.nn.functional as F
from layers.module import Ushape


#注意这个fmf输出是输入通道数的二倍
class FMF(nn.Module):
    def __init__(self, in_channels, out_channels, depth=5):
        super(FMF, self).__init__()
        self.u = Ushape.UNet(out_channels, in_channels + 15, depth=depth, merge_mode='concat')
        self.sig = nn.Sigmoid()

        self.conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        #第一个featurefusion其中heatmap缩放用的maxpooling，128到32
    def forward(self, featuremap, heatmap):

        heatmapall = F.interpolate(heatmap, featuremap.shape[2:4])

        #print(heatmapall.shape)
        #print(featuremap.shape)
        x = torch.cat([heatmapall,featuremap],1)
        #print(x.shape)
        # x = np.c_[heatmapall, featuremap]
        out = self.u(x)
        out = self.sig(out)
        out = out * featuremap
        #print(out.shape)
        #out = np.c_[out, featuremap]
        out = torch.cat([out,featuremap],1)
        out = self.conv(out)
        return out


