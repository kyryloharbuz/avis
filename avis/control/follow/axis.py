# THE BASE CONTROL AXIS -- the mechanics shared by yaw, vertical and distance.
#
# The deadzone used to be implemented THREE different ways: by scaling for yaw,
# by a hard cut for vertical, by subtraction for distance. One and the same
# intent, three implementations, three sets of bugs.
#
# Here it exists once, and in the mathematically correct form: we SUBTRACT the
# zone width from the error. The consequence is that at the zone boundary the
# command is exactly ZERO and grows smoothly from there. There is no "step" that
# made the drone jerk the moment the target drifted slightly out of the zone.

import math

from avis.control.controller import PID


class Axis:
    """One axis: error -> soft deadzone -> PID -> command."""

    # NOTE ON THE NAME. ki defaults to 0 and no axis currently overrides it, so
    # in practice all three axes run as PD controllers, not full PID. That is
    # intentional -- an integral term on a Wi-Fi-delayed loop with frequent
    # target losses mostly accumulates windup -- but call it PD when describing
    # the system, not PID.
    def __init__(self, kp, kd, deadzone, ki=0.0):
        self._pid = PID(kp=kp, ki=ki, kd=kd)
        self._deadzone = deadzone

    @property
    def deadzone(self):
        return self._deadzone

    def shape(self, error):
        """Soft deadzone: exactly 0 inside, growing smoothly from zero outside.
        The units of the error and of the zone must match (px with px, metres
        with metres)."""
        if abs(error) < self._deadzone:
            return 0.0
        return error - math.copysign(self._deadzone, error)

    def update(self, error, dt):
        """Compute the command. Inside the deadzone we do not touch the PID at
        all, so that its derivative does not accumulate noise from tiny box
        wobbles."""
        shaped = self.shape(error)
        if shaped == 0.0:
            return 0.0
        return self._pid.update(shaped, dt)

    def reset(self):
        self._pid.reset()

    def hold(self):
        """Command 0, but with a PID reset -- for when the axis is temporarily
        untrustworthy. Kept separate from reset() so that the INTENT is visible
        at the call site: not "start over" but "this axis cannot be trusted right
        now"."""
        self._pid.reset()
        return 0.0
