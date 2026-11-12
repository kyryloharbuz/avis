# YAW -- the simplest axis: keep the target horizontally centred in the frame.
#
# Works in PIXELS (both the error and the deadzone). Deliberately so: the kp
# coefficient stays intuitive ("command per pixel") and units do not mix.

from avis.control.follow.axis import Axis


class YawAxis(Axis):
    """Brings the target to the frame centre by rotating the drone."""

    def __init__(self, cfg):
        super().__init__(kp=cfg.yaw_kp, kd=cfg.yaw_kd,
                         deadzone=cfg.yaw_deadzone_px)

    def command(self, error, dt):
        """error.error_x in pixels: right of centre -> positive, so turn right."""
        return self.update(error.error_x, dt)
