"""models package — exports for AgroKD-Net and OJS losses."""

from .agrokd_net import AgroKDNet
from .ojs_loss import OJSTotalLoss

__all__ = ["AgroKDNet", "OJSTotalLoss"]
