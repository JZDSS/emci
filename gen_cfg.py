from google.protobuf import text_format
from proto.all_pb2 import *

cfg = Config(
    root='./test1',
    lr=LearningRate(
        type=LearningRate.PIECEWISE,
        piecewise_constant=PiecewiseConstant(
            boundary=[50000, 100000],
            value=[2e-5, 4e-6, 8e-7]
        )
    ),
    loss=Loss(
        type=Loss.WING2,
        wing_loss2=WingLoss2(
            w1=5,
            epsilon1=2,
            w2=10,
            epsilon2=2
        )
    ),
    device=GPU
)

s = text_format.MessageToString(cfg)
print(s)