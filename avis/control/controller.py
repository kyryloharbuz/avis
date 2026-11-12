# PID -- the basic building block of the controllers.
#
# This file also used to hold a 250-line FollowController with five
# responsibilities. That has been broken out into the avis/control/follow/
# package, where each axis is its own class with its own tests. What remains
# here is the PID maths itself.
#
# PID in three lines (steering like a driver):
#   P -- react proportionally to the current error (farther -> harder).
#   I -- accumulate a small persistent error over time and squeeze it to zero.
#   D -- watch HOW FAST the error is changing and brake in advance (damping).
# Output = P + I + D.


def _clamp(value, lo, hi):                  # a "clamp" into range (there is no builtin)
    return max(lo, min(value, hi))


# PID is an INTERNAL building block (not public API). That is why it is absent
# from the package __init__.py: only FollowController uses it (composition).
class PID:
    """A PID controller for ONE quantity."""

    # output_limit is worth knowing about when reading measurements of this
    # system: the output is clamped to +/-100 (the Tello RC range) INSIDE the
    # controller, before any supervisor limit. So "the PID asked for 100" always
    # means "the PID hit its own ceiling", and a linearity check must probe
    # errors small enough that both samples stay below it.
    def __init__(self, kp, ki, kd, output_limit=100.0, integral_limit=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._output_limit = output_limit       # output limit (Tello RC range)
        self._integral_limit = integral_limit   # protection against integral windup
        self._integral = 0.0                     # accumulated sum (I)
        self._prev_error = None                  # previous error (for D)

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def update(self, error, dt):
        if dt <= 0:                              # guard against division by 0 in the D term
            return 0.0

        p = self.kp * error                      # --- P ---

        self._integral += error * dt             # --- I ---
        self._integral = _clamp(self._integral, -self._integral_limit, self._integral_limit)
        i = self.ki * self._integral

        if self._prev_error is None:             # --- D ---
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        d = self.kd * derivative
        self._prev_error = error

        return _clamp(p + i + d, -self._output_limit, self._output_limit)
