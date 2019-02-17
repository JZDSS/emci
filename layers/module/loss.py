from layers.module import wing_loss, wing_loss2
from proto import all_pb2

def get_criterion(cfg):
    if cfg.type == all_pb2.Loss.WING:
        return wing_loss.WingLoss(cfg.wing_loss.w, cfg.wing_loss.epsilon)
    elif cfg.type == all_pb2.Loss.WING2:
        return wing_loss2.WingLoss2(cfg.wing_loss2.w1,
                                    cfg.wing_loss2.epsilon1,
                                    cfg.wing_loss2.w2,
                                    cfg.wing_loss2.epsilon2)
