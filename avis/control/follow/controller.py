# The tracking COORDINATOR.
#
# This class used to be 250 lines with 20 parameters, combining five
# responsibilities: horizontal, vertical, distance (in TWO incompatible
# strategies), deadzones, level adaptation, gates and axis decoupling. Every
# change there broke something else -- confirmed in practice.
#
# Now it does exactly one thing: it COORDINATES independent axes. All the maths
# lives in them; what remains here is the call order and the coupling between
# axes:
#   * distance exposes the RATE of its own change -> vertical decides whether to
#     trust itself;
#   * yaw determines ALIGNMENT -> alignment permits or suppresses forward motion.

from avis.control.follow.alignment import AlignmentGate
from avis.control.follow.config import FollowConfig
from avis.control.follow.distance import DistanceAxis
from avis.control.follow.vertical import VerticalAxis
from avis.control.follow.yaw import YawAxis
from avis.models import Command, ControlError


class FollowController:
    """Reduces three independent axes to a single motion command."""

    def __init__(self, config: FollowConfig = None, **overrides):
        # We accept both a ready-made config and spot overrides:
        # FollowController(yaw_kp=0.5). The latter is handy for replay tuning
        # when you are sweeping a single parameter.
        self.config = config or FollowConfig(**overrides)
        self._yaw = YawAxis(self.config)
        self._vertical = VerticalAxis(self.config)
        self._distance = DistanceAxis(self.config)
        self._gate = AlignmentGate(self.config)

    def reset(self):
        """Reset on target change or loss: every axis forgets its accumulated state.

        WORTH KNOWING WHEN YOU READ THE CALL SITE. The pipeline calls this on
        every target-id change, and ByteTrack reissues ids far more often than
        you would guess: 39 times across one 218-second flight
        (flight_20261002_185914, 25 distinct ids). So the PID history is wiped
        39 times per flight. Resetting the Kalman filter there is clearly right
        -- the position may have jumped -- but wiping the yaw PID is more
        debatable, since the new box is usually in roughly the same place. It
        shows up as command jitter."""
        self._yaw.reset()
        self._vertical.reset()
        self._distance.reset()

    def update(self, error: ControlError, dt) -> Command:
        """Compute the motion command from the current error.

        THERE USED TO BE A trust_distance FLAG HERE, and in PRED mode (target
        occluded, steering by the Kalman prediction) forward motion was forced to
        zero. The intent was right -- do not fly on STALE data -- but the
        implementation was crude: instead of bounding how long we trust the data,
        it simply stopped dead.

        Measured on flight_20260926_212115 (2587 AUTO frames):
            frames with no target box : 25%
            forward motion meanwhile  : zero in 100% of cases
            75 episodes, MEDIAN 0.18 s, longest 9.2 s
        So a quarter of the time the drone stood still, and mostly because of
        fifth-of-a-second pauses over which the distance physically does not
        change. Hence the second complaint -- "it stays far away": the approach
        was interrupted every few frames.

        Now ONE place makes that decision -- the distance axis itself. It already
        knows how to hold the last trustworthy measurement for a bounded time
        (distance_hold_s), and once that time runs out it stops on its own. The
        controller does not need to know about tracker states."""
        # The vertical axis needs the rate of distance change -- compute it
        # ALWAYS, otherwise the derivative lies after every pause.
        self._distance.track_rate(error.distance_m, dt)

        yaw = self._yaw.command(error, dt)
        vertical = self._vertical.command(error, dt, self._distance.rate)
        # Forward motion is permitted to the extent the drone already faces the target.
        forward = self._distance.command(error, dt) * self._gate.factor(error.norm_x)

        return Command(yaw=yaw, vertical=vertical, forward=forward)

    # -- For metrics and reports ---------------------------------------
    @property
    def distance_target_m(self):
        """The distance setpoint. The SINGLE source of truth -- the metric reads
        it from here so that it measures the same thing the system controls
        (the two used to diverge)."""
        return self.config.distance_target_m

    @property
    def distance_deadzone_m(self):
        return self.config.distance_deadzone_m
