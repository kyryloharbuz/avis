# ПОВОРОТ — найпростіша вісь: тримати ціль по горизонталі в центрі кадру.
#
# Працює в ПІКСЕЛЯХ (і помилка, і зона спокою). Це навмисно: коефіцієнт kp
# лишається інтуїтивним ("команда на піксель"), а одиниці не змішуються.

from avis.control.follow.axis import Axis


class YawAxis(Axis):
    """Доводить ціль до центру кадру поворотом дрона."""

    def __init__(self, cfg):
        super().__init__(kp=cfg.yaw_kp, kd=cfg.yaw_kd,
                         deadzone=cfg.yaw_deadzone_px)

    def command(self, error, dt):
        """error.error_x у пікселях: правіше центру → +, отже поворот праворуч."""
        return self.update(error.error_x, dt)
