# DISTANCE -- the axis that holds a set distance to the target. Works IN METRES.
#
# WHAT WAS REMOVED FROM HERE, AND WHY. This axis used to receive a distance that
# DistanceEstimator measured with TWO measures and automatic switching: box
# height while the box was intact, and shoulder width once the full stature no
# longer fitted in frame. Measurements on flight_20260926_224656 showed the
# design cost more than it gave:
#   * 56% of measurements went through width, and width depends on POSE -- a
#     body turning sideways read as "the target moved away", and the drone flew
#     at the person;
#   * 243 measure switches, median jump 0.18 m -- LARGER than the deadzone.
# Now there is one measure (height), and half the complexity here became
# unnecessary: the median window shrank from 5 frames to 3, because the
# pose-driven outliers left along with width. Measured on the same recording:
# frame-to-frame |delta distance| fell from 0.019 to 0.009 m (median) and from
# 0.114 to 0.058 m (p90).
#
# A NOTE ON THE POSE FIGURES. The height/width ratio range quoted elsewhere in
# this project as "1.61 to 2.77 against a fixed 2.26" is a percentile band
# (roughly p16-p95), not the full range. Re-measured over the 2872 frames with a
# target in AUTO on that flight, the ratio actually spans 0.67 to 4.64 with a
# median of 2.15. The real spread is therefore wider than the old comment
# implied, which only strengthens the argument for dropping width.
#
# WHY METRES AND NOT PIXELS OF HEIGHT. Height ~ 1/distance, so in pixels the
# error is nonlinear: "far" would yield enormous numbers and "close" tiny ones,
# making retreat inherently weaker than approach. In metres the response is
# symmetric.
#
# THE MEASUREMENT LIMIT AND WHY RETREAT NOW WORKS. Height saturates once the
# person no longer fits in the frame (nearest_measurable_m; for 478 px that is
# 1.33 m). The setpoint sits BEYOND that limit with margin, so a saturated
# measurement always reads as "too close" -- and the axis commands a retreat.
# The setpoint used to be 0.75 m, on the FAR SIDE of the limit, so the same
# saturation read as "too far": 87% of motor commands were "forward" even with
# the person standing right against the drone. The invariant is checked in
# FollowConfig.check_measurable().

from collections import deque

from avis.control.follow.axis import Axis


class DistanceAxis(Axis):
    """Holds a set distance to the target. Works EXCLUSIVELY in metres."""

    def __init__(self, cfg):
        super().__init__(kp=cfg.distance_kp, kd=cfg.distance_kd,
                         deadzone=cfg.distance_deadzone_m)
        self._target = cfg.distance_target_m
        self._retreat_factor = cfg.retreat_factor
        self._prev_distance = None
        self._rate = 0.0             # m/s, needed by the vertical axis for freezing

        # THE FRAME-FRACTION GUARD -- the last line of defence.
        # It is needed precisely because the measurement has a LIMIT: beyond
        # nearest_measurable_m the height stops growing, and the distance can no
        # longer say how close we actually are. The frame fraction keeps growing,
        # so it alone can tell "close" from "point-blank". It is not
        # pose-invariant (arms spread raise it), but for a "stop" decision that
        # is sufficient -- it can only err in the safe direction.
        self._fill_stop = cfg.frame_fill_stop        # above this, forward is forbidden
        self._fill_retreat = cfg.frame_fill_retreat  # above this, forced retreat
        self._retreat_cmd = cfg.frame_fill_retreat_command

        # A MEDIAN against single box outliers. A median specifically, not a mean:
        # it discards a bad measurement rather than smearing it. Measured: 1.4%
        # of frames produce a jump larger than the deadzone (up to 0.69 m) --
        # without the filter the drone would twitch.
        self._window = deque(maxlen=cfg.distance_median_frames)

        # HOLDING THE LAST MEASUREMENT when there is no box at all.
        #
        # Needed for the PRED state: the target is occluded, we steer by the
        # Kalman prediction, there is no box -- and therefore no distance.
        # Forward motion used to drop to exactly zero at that moment, and the
        # measurement was this (flight_20260926_212115, 2587 AUTO frames): 25% of
        # frames had no box, forward motion was zero in 100% of those, across 75
        # episodes with a MEDIAN of 0.18 s. So a quarter of the time the drone
        # stood still -- and almost always because of fifth-of-a-second pauses
        # over which the distance physically does not change. That was the source
        # of the "it keeps too far away" complaint.
        #
        # "I cannot measure" and "stop" are different things, just as "I see
        # nothing" and "the target is gone" are. So we hold the last trustworthy
        # measurement briefly, and once the time runs out we stop on our own.
        self._hold_seconds = cfg.distance_hold_s
        self._held = None            # last trustworthy distance
        self._held_age = 0.0         # how long we have been using it

    @property
    def rate(self):
        """Rate of change of distance (m/s). The vertical axis uses it to tell
        'the target crouched' from 'the drone closed in'."""
        return self._rate

    def reset(self):
        super().reset()
        self._prev_distance = None
        self._rate = 0.0
        self._window.clear()
        self._held = None
        self._held_age = 0.0

    def smooth(self, distance_m, dt=0.0):
        """Median of recent measurements -- damps single box outliers.

        If there is no measurement we return the last trustworthy one for a
        short while rather than None (see the constructor comment)."""
        if distance_m is None:
            if self._held is None:
                return None
            self._held_age += dt
            if self._held_age > self._hold_seconds:
                return None          # held too long -- beyond this it is fiction
            return self._held

        self._window.append(distance_m)
        ordered = sorted(self._window)
        median = ordered[len(ordered) // 2]
        self._held = median          # remember it in case the measurement is lost
        self._held_age = 0.0
        return median

    def track_rate(self, distance_m, dt):
        """Update the estimate of how fast the distance is changing. Call EVERY
        FRAME, even when no command is computed -- otherwise the derivative
        lies."""
        if distance_m is None or dt <= 0:
            self._rate = 0.0
            return
        if self._prev_distance is not None:
            raw = (distance_m - self._prev_distance) / dt
            self._rate = 0.7 * self._rate + 0.3 * raw   # smoothed: 5-7 FPS is noisy
        self._prev_distance = distance_m

    def command(self, error, dt):
        """Error = current distance - desired. Far -> positive, i.e. forward."""
        # THE GUARD COMES FIRST, before any computation: if the box occupies too
        # much of the frame we are close REGARDLESS of what the distance says.
        # This is the path that saves us beyond the measurement limit.
        if error.frame_fill >= self._fill_retreat:
            self._pid.reset()
            # The magnitude is FIXED and deliberately above the drone's
            # sensitivity threshold: a guard the drone does not execute is worse
            # than no guard at all.
            return -abs(self._retreat_cmd)

        distance = self.smooth(error.distance_m, dt)
        if distance is None:
            return self.hold()       # no measurement, and nothing left to hold

        out = self.update(distance - self._target, dt)

        # RETREAT is slower than approach: the drone sees nothing behind itself.
        #
        # Note that BEYOND THE MEASUREMENT LIMIT the retreat comes out nearly
        # constant: the measurement saturates at nearest_measurable_m, so the
        # error stops growing and there is nowhere for proportionality to come
        # from. That is fine -- responding proportionally to something you cannot
        # see is impossible anyway; past that point the frame-fraction guard
        # takes over, and it has no limit.
        if out < 0:
            out *= self._retreat_factor

        # FORWARD motion is forbidden by FRAME FRACTION -- a direct proximity
        # signal. Retreat passes through freely: only approach needs clamping.
        if error.frame_fill >= self._fill_stop:
            return min(out, 0.0)
        return out
