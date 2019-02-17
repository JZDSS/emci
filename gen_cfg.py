from google.protobuf import text_format
from proto.all_pb2 import *

cfg = Config(
    root='./exp/wing2(5,13)-align-j3-m0_2_0_15',
    lr=LearningRate(
        type=LearningRate.PIECEWISE,
        piecewise_constant=PiecewiseConstant(
            boundary=[100000, 150000],
            value=[2e-5, 4e-6, 8e-7]
        )
    ),
    weight_decay=5e-4,
    batch_size=16,
    max_iter = 2000000,
    loss=Loss(
        type=Loss.WING2,
        wing_loss2=WingLoss2(
            w1=5,
            epsilon1=2,
            w2=13,
            epsilon2=2
        )
    ),
    device=GPU
)

s = text_format.MessageToString(cfg)
print(s)