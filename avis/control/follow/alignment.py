# COUPLING BETWEEN YAW AND FORWARD MOTION.
#
# The drone moves forward ALONG ITS OWN NOSE. If the target is off to the side
# and we apply full forward travel, we fly past it on an arc and -- worst of all
# -- lose it from the frame. So forward motion is permitted exactly to the
# extent that we already face the target.
#
# This is NOT a hard sequence ("turn first, THEN go") and not a blind
# "simultaneously" either -- it is a smooth blend: yaw always works, and forward
# travel grows as alignment improves.

class AlignmentGate:
    """A 0..1 multiplier for forward motion based on how centred the target is."""

    def __init__(self, cfg):
        self._full = cfg.align_full   # within this offset, full travel
        self._stop = cfg.align_stop   # beyond this, forward motion forbidden

    def factor(self, norm_x) -> float:
        """norm_x -- offset in fractions of the frame half-width (0 = centre, 1 = edge)."""
        offset = abs(norm_x)
        if offset <= self._full:
            return 1.0
        if offset >= self._stop:
            return 0.0
        return (self._stop - offset) / (self._stop - self._full)
