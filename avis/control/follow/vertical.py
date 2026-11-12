# VERTICAL -- the trickiest axis, and here is why.
#
# PROBLEM 1 (measured across 10 flights): the centre of a person's box sits
# systematically ABOVE the centre of the frame (y ~= 230 against 360). If the
# target is the frame centre, the drone receives an "up" command CONTINUOUSLY
# and creeps towards the ceiling.
#   -> Solution: aim not at the frame centre but at the target's OWN USUAL
#      LEVEL, learned with a slow moving average. A constant offset is absorbed
#      and stops producing a command; a fast change (crouching, standing up) is
#      not.
#
# PROBLEM 2 (also measured): when the drone CLOSES IN, the box centre drifts
# downwards by ~56 px ON ITS OWN -- pure geometry, not target motion. The
# adaptive level read that as "she crouched" and descended.
#   -> Solution: while the DISTANCE IS CHANGING faster than a threshold, FREEZE
#      the axis. Crouching happens WITHOUT a distance change, so it still gets
#      through.

from avis.control.follow.axis import Axis


class VerticalAxis(Axis):
    """Holds the target at its usual level; reacts to crouching, not to closing in."""

    def __init__(self, cfg):
        super().__init__(kp=cfg.vertical_kp, kd=cfg.vertical_kd,
                         deadzone=cfg.vertical_deadzone_px)
        self._enabled = cfg.vertical_enabled
        self._tau = cfg.vertical_adapt_tau_s
        self._freeze_speed = cfg.vertical_freeze_speed_mps
        self._reference = None       # the target's usual level (px of error from centre)

    def reset(self):
        super().reset()
        self._reference = None       # the level belongs to a SPECIFIC target

    def frozen(self, error, distance_rate) -> bool:
        """Whether the vertical geometry can be trusted RIGHT NOW."""
        if error.box_clipped:
            return True              # box clipped by an edge -- the centre is unreliable
        return abs(distance_rate) > self._freeze_speed

    def command(self, error, dt, distance_rate):
        if not self._enabled:
            # Altitude is held by the Tello itself (see the config comment).
            # WORTH KNOWING WHEN DEBUGGING ALTITUDE: with this disabled the
            # vertical command is identically zero -- verified over all 1933
            # AUTO frames of flight_20261002_185914. So any altitude change you
            # observe in flight does not come from this controller; look at
            # sustained max-rate yaw and high forward pitch instead, both of
            # which upset the Tello's own height hold.
            return 0.0
        if self.frozen(error, distance_rate):
            self._pid.reset()        # neither learn nor move on untrustworthy data
            return 0.0

        if self._reference is None:
            self._reference = error.error_y
            return 0.0

        deviation = error.error_y - self._reference
        # Slowly pull the level towards the current value. The y axis points
        # DOWN, so we invert the sign: target below its usual level -> fly DOWN.
        a = min(1.0, dt / self._tau)
        self._reference = (1 - a) * self._reference + a * error.error_y
        return -self.update(deviation, dt)
