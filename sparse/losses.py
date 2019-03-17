from layers.module import wing_loss, wing_loss2, pose_loss, l2_loss, boundary_loss
from proto import all_pb2

def get_criterion(cfg):
    if cfg.type == all_pb2.Loss.WING:
        return wing_loss.WingLoss(cfg.wing_loss.w, cfg.wing_loss.epsilon)
    elif cfg.type == all_pb2.Loss.WING2:
        return wing_loss2.WingLoss2(cfg.wing_loss2.w1,
                                    cfg.wing_loss2.epsilon1,
                                    cfg.wing_loss2.w2,
                                    cfg.wing_loss2.epsilon2)
    elif cfg.type == all_pb2.Loss.POSE:
        cfg = cfg.pose_loss
        return pose_loss.PoseLoss(cfg.max_w1, cfg.min_w1,
                                  cfg.max_epsilon1, cfg.min_epsilon1,
                                  cfg.max_w2, cfg.min_w2,
                                  cfg.max_epsilon2, cfg.min_epsilon2)
    elif cfg.type == all_pb2.Loss.L2:
        return l2_loss.L2Loss()
    elif cfg.type == all_pb2.Loss.BOUNDARY:
        cfg = cfg.boundary_loss
        if cfg.version == all_pb2.BoundaryLoss.SOFT:
            return boundary_loss.BoundaryLossN(version='soft',
                                               alpha=cfg.alpha,
                                               threshold=cfg.threshold,
                                               threshold_decay=cfg.threshold_decay)
        elif cfg.version == all_pb2.BoundaryLoss.HARD:
            return boundary_loss.BoundaryLossN(version='hard',
                                               threshold=cfg.threshold,
                                               threshold_decay=cfg.threshold_decay)
