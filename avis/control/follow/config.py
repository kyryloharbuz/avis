# TRACKING CONFIGURATION -- every parameter in ONE place, with validation.
#
# Why a separate class: FollowController used to take 20 constructor arguments
# with no checking at all. That produced real silent breakage -- for instance a
# sensitivity floor of 20 overriding a ceiling of 15, so the command came out
# LARGER than permitted. The config catches that immediately.
#
# UNITS ARE IN EVERY NAME. Deliberately: a previous version kept pixels,
# fractions of the frame, metres and area in px^2 inside one class, and
# confusing them was a source of bugs.

from dataclasses import dataclass


@dataclass
class FollowConfig:
    """Tracking parameters. Every field carries its unit in the name."""

    # -- YAW (horizontal) ----------------------------------------------
    # Command per PIXEL of error. 0.50 -> 0.75.
    # At 0.50 with a ceiling of 70, full travel was only reached at |ex| ~= 164
    # px -- half of the frame half-width, meaning the target had to drift almost
    # to the edge before the drone turned at full rate. Measured in flight, the
    # median |yaw| was about 28-34 depending on how you count (32 over frames
    # with a target and a non-zero command, 25 if zeros are included).
    # At 0.75 with a ceiling of 90, full travel arrives at |ex| ~= 144 px.
    #
    # BE AWARE OF THE SIDE EFFECT. Measured by replaying flight_20261002_185914:
    # the yaw command now sits at the ceiling in 56% of frames with a command.
    # That is deliberate for a target walking past, but it does mean yaw control
    # is closer to bang-bang than proportional in that regime, and sustained
    # max-rate rotation starves the Tello's downward vision positioning of
    # optical flow, which shows up as altitude drift.
    yaw_kp: float = 0.75
    yaw_kd: float = 0.10              # damping
    yaw_deadzone_px: float = 24.0     # closer to centre than this -- do not move

    # -- VERTICAL ------------------------------------------------------
    # DISABLED BY DEFAULT. The reason is a FEEDBACK LOOP I had not anticipated in
    # the adaptive-level design:
    #   drone climbs -> target appears LOWER in frame -> error -> command down
    #   -> drone descends -> target appears HIGHER -> command up -> and round.
    # On its own this is negative feedback (self-correcting), but with ~250 ms of
    # Wi-Fi latency and slow level learning (tau 8 s) it turns into OSCILLATION.
    # Which is exactly what was observed: "it rose and fell" with a level target.
    #
    # Why disabling is safe: the Tello holds altitude by itself, and the vertical
    # axis contributes least to tracking quality -- measurements showed a small
    # |ey| even without it. (An earlier revision claimed ey ~= 6 px; re-measured
    # over the recordings the median |ey| is 11-15 px. Still small relative to
    # the 45 px deadzone, so the conclusion stands, but the number was optimistic.)
    # To re-enable: FollowController(vertical_enabled=True).
    vertical_enabled: bool = False
    vertical_kp: float = 0.35
    vertical_kd: float = 0.045
    vertical_deadzone_px: float = 45.0
    # How slowly the target's "usual level" is learned. A large tau means we
    # ignore a constant box offset but still react to someone crouching quickly.
    vertical_adapt_tau_s: float = 8.0
    # If the distance changes faster than this we FREEZE the vertical axis: while
    # closing in, the box centre drifts downwards by itself (measured ~56 px),
    # and that is not the target moving.
    vertical_freeze_speed_mps: float = 0.35

    # -- DISTANCE ------------------------------------------------------
    # THE SETPOINT MUST LIE BEYOND THE MEASUREMENT LIMIT -- the primary
    # requirement of this axis, and violating it cost a whole flight.
    #
    # Distance is measured from box height, and height saturates as soon as the
    # person stops fitting in the frame: for 478 px that is 1.33 m (see
    # distance.nearest_measurable_m). The setpoint used to be 0.75 m -- on the
    # FAR SIDE of that limit, in a region the camera physically cannot resolve.
    # The consequence, measured on flight_20260926_224656: the distance read
    # "too far" in essentially every frame with a target, 87% of commands sent
    # to the motors were "forward" and only 2% "backward", and even with the
    # person standing right against it the drone pushed on -- held back only by
    # the frame-fill guard. The operator described it as "it will not back off,
    # it just freezes" and "instead of turning in place it flies".
    #
    # ON THE "93%" THAT USED TO BE QUOTED HERE. The 87%/2% command split
    # reproduces exactly from the log. The "93% of frames read too far" figure
    # came from the old dual-measure DistanceEstimator, which has been deleted;
    # with today's height-only code the figure is 100% by construction, since the
    # 1.33 m measurement floor already exceeds the old 0.95 m threshold. Either
    # way the point is the same, and the invariant below is what prevents it.
    #
    # 1.70 m gives a 374 px box (78% of the frame) and 0.17 m of margin from the
    # near edge of the deadzone to the measurement limit -- all three verified.
    # check_measurable() enforces the invariant.
    distance_target_m: float = 1.70
    # Command per METRE of error. 30 -> 55 -> 95.
    #
    # WHY 30 WAS TOO LITTLE. The computed profile at setpoint 1.70, deadzone 0.20:
    #     2.0 m -> 5     2.5 m -> 18     3.0 m -> 33     4.0 m -> 63
    # The first two are BELOW the drone's sensitivity threshold (min_forward=20),
    # so in the 1.9-2.6 m band -- exactly the band the operator walks in --
    # motion was governed not by the controller but by the lift to the floor. The
    # drone crawled at 20 (~0.2 m/s) whether the target was one metre or one and
    # a half beyond the deadzone. Meanwhile the ceiling (55) was barely used: to
    # reach it the target had to be 4 m away.
    #
    # WHY 55 WAS ALSO TOO LITTLE. The same disease, milder. Measured on
    # flight_20261002_185914 (471 frames with a motor command): the forward axis
    # was clamped by the RAMP in 0% of frames, and by the ceiling of 80 in only
    # 12%. So acceleration was limited neither by smoothing nor by the ceiling,
    # but by the GAIN. The profile at 55:
    #     2.0 m -> 20 (floor)   2.2 m -> 20 (floor)   2.5 m -> 33   3.0 m -> 61
    # A person walks at 1.2-1.4 m/s, while the drone in its working band of
    # 2.0-2.5 m had 20-33 of the available 80 (~0.2-0.35 m/s) and only reached
    # full speed at 3.5 m -- i.e. it caught up only once hopelessly behind.
    #
    # At 95 the profile becomes: 2.2 m -> 29, 2.4 m -> 47, 2.43 m -> 50 (ceiling).
    # Note the ceiling here is the CURRENT max_forward of 50, not the 80 of the
    # measurement above: the same tuning round that raised this gain lowered that
    # ceiling, because a hard forward pitch smears the frame and costs detections.
    #
    # WHY THIS DOES NOT BREAK BRAKING. The deadzone must be wider than the
    # braking distance, and what decides that is neither the ceiling nor the
    # gain, but the speed of ENTRY into the zone. Entry still happens at
    # min_forward=20 (braking distance ~0.11 m against a 0.20 m zone), because
    # 0.05 m beyond the zone edge the error yields 0.05 x 95 = 5 and the
    # sensitivity floor lifts the command anyway. Verified for kp 55, 95 and even
    # 140: the entry speed does NOT depend on kp at all. High speeds only occur
    # far out, where there is room to brake.
    distance_kp: float = 95.0
    distance_kd: float = 2.0
    # THE DEADZONE -- a compromise between two opposing requirements.
    # Too wide (it was 0.25) gives a comfortable 1.35-1.85 m band in which the
    # drone "correctly" almost never moves. Too narrow and the drone physically
    # cannot stop in time and overshoots the target, because the zone must be
    # WIDER than the braking distance. Braking distance measured: at command 20
    # the drone travels another ~0.11 m after "stop", at 55 about ~0.30 m. Hence
    # 0.20 m.
    distance_deadzone_m: float = 0.20
    # How many recent measurements to average with a MEDIAN.
    # It was 5 back when distance was also measured from shoulder width and body
    # rotation produced large outliers. With a single measure (height) the noise
    # halved -- measured frame-to-frame |delta| 0.009 m against 0.019 m -- so 3
    # frames are enough, and a shorter window also lags less.
    distance_median_frames: int = 3
    # How many seconds to hold the LAST trustworthy distance when the measurement
    # disappears. The measurement vanishes when the box is clipped both at the
    # top and at the side -- i.e. the target has simply drifted to the frame
    # edge. That lasts fractions of a second while yaw brings it back to centre,
    # and over that time the distance physically barely changes.
    # Forward motion used to collapse to zero at that moment: measured, at |ex|
    # of 150-250 px the axis switched off in 75% of frames, and the drone never
    # got closer than about 1.4 m. Safety is held by frame_fill, which depends
    # neither on pose nor on clipping.
    distance_hold_s: float = 1.0

    # -- PROXIMITY GUARD (fraction of the frame) -----------------------
    # Works BEYOND THE MEASUREMENT LIMIT -- where box height has already
    # saturated and nothing else can say how close we actually are. The frame
    # fraction keeps growing past that point.
    #
    # Measured on flight_20260926_224656 (2656 height measurements):
    #     1.50 m -> 27% of frame      2.00 m -> 15%
    #     1.75 m -> 19%               2.50 m ->  9%
    # So both thresholds below sit DEEP inside the region where the measurement
    # is already blind (~1.1 m and ~0.95 m), and in normal operation at 1.70 m
    # they are never touched.
    #
    # The frame fraction is NOT pose-invariant (arms spread raise it without the
    # distance changing) -- which is why it is trusted only with "stop / back
    # off" decisions. It can only err in the safe direction, and it never makes
    # the "chase" decision at all.
    frame_fill_stop: float = 0.50      # above this, forward motion is FORBIDDEN (~1.1 m)
    frame_fill_retreat: float = 0.65   # above this, FORCED retreat (~0.95 m)
    # The size of the forced retreat, chosen above the drone's sensitivity
    # threshold (20): a guard whose command the drone does not execute is worse
    # than no guard, because it creates a false sense of safety.
    #
    # BUT IT DOES NOT REACH THE MOTORS AT THIS SIZE. FollowController multiplies
    # the whole distance axis by AlignmentGate, retreat included, so the further
    # the target drifts off centre the weaker the retreat becomes: measured 25 in
    # the centre, 9.7 at norm_x 0.63, and exactly 0 beyond align_stop. The
    # supervisor's sensitivity floor happens to lift the middle of that range
    # back to 20, but at the frame edge the guard is silent -- the drone will
    # stop approaching, yet it will not back away. Gating only positive forward
    # motion would fix it; that is a control change, not a comment.
    frame_fill_retreat_command: float = 25.0
    # Retreat is slower than approach: the drone sees nothing behind itself.
    retreat_factor: float = 0.5

    # -- COUPLING BETWEEN YAW AND FORWARD MOTION -----------------------
    # Fractions of the frame HALF-WIDTH (0 = centre, 1 = edge).
    align_full: float = 0.35          # within this, full forward travel
    align_stop: float = 0.80          # beyond this, forward motion forbidden

    def __post_init__(self):
        """Checks that did not exist before. Every one of them caught a real bug."""
        if self.distance_target_m <= 0:
            raise ValueError("distance_target_m must be positive")
        if self.distance_deadzone_m < 0:
            raise ValueError("distance_deadzone_m cannot be negative")
        if not 0 < self.frame_fill_stop < self.frame_fill_retreat <= 1.0:
            raise ValueError(
                "must have 0 < frame_fill_stop < frame_fill_retreat <= 1")
        if self.distance_median_frames < 1:
            raise ValueError("distance_median_frames must be >= 1")
        if self.distance_hold_s < 0:
            raise ValueError("distance_hold_s cannot be negative")
        if self.distance_deadzone_m >= self.distance_target_m:
            raise ValueError(
                "the deadzone is wider than the setpoint itself -- the drone would never move")
        if not 0 <= self.align_full < self.align_stop <= 1.0:
            raise ValueError(
                "must have 0 <= align_full < align_stop <= 1 "
                f"(got {self.align_full} and {self.align_stop})")
        if not 0 < self.retreat_factor <= 1.0:
            raise ValueError("retreat_factor must be within (0, 1]")
        for name in ("yaw_deadzone_px", "vertical_deadzone_px"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.vertical_adapt_tau_s <= 0:
            raise ValueError("vertical_adapt_tau_s must be positive")

    # -- The check that cannot be done in __post_init__ ----------------
    def check_measurable(self, frame_h):
        """Whether the distance setpoint is REACHABLE at all for this frame size.

        WHY A SEPARATE METHOD. This is the most expensive mistake available here,
        but it cannot be checked in the constructor: the frame height is only
        known once the first frame has arrived, and the config is built long
        before that.

        THE INVARIANT. Box height measures distance only while the person fits in
        the frame; closer in, the measurement saturates at
        nearest_measurable_m(frame_h). For a saturated measurement to read as
        "too close" (rather than "too far"), the limit must lie BELOW the near
        edge of the deadzone:

            K / frame_h  <  distance_target_m - distance_deadzone_m

        Violating this inequality is precisely the bug that produced a flight
        where the distance read "too far" in essentially every frame and 87% of
        motor commands were "forward" at a setpoint of 0.75 m: the drone chased a
        person standing right in front of its nose.

        Returns a warning string, or None if all is well. It does not raise: in
        the air it is better to fly with a warning than to fall out of the sky
        because of a ValueError on the first frame."""
        from avis.distance import nearest_measurable_m

        floor = nearest_measurable_m(frame_h)
        if floor is None:
            return None
        near_edge = self.distance_target_m - self.distance_deadzone_m
        if floor < near_edge:
            return None
        return (
            f"DISTANCE SETPOINT UNREACHABLE: box height saturates at {floor:.2f} m "
            f"(frame {frame_h} px), while the near edge of the deadzone is {near_edge:.2f} m. "
            f"Closer than {floor:.2f} m the distance freezes and reads as "
            f"'too far', so the drone will push FORWARD even at point-blank range. "
            f"Set distance_target_m > {floor + self.distance_deadzone_m:.2f} m."
        )
