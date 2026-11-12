# TRACKING QUALITY METRICS in the style of the Andon Labs Drone-Bench.
#
# Why: until now we measured FPS -- a metric of SPEED, not QUALITY. FPS says
# nothing about whether the drone actually holds the target. Andon Labs measure
# quality in their benchmark (the "Follow" task) and publish a human baseline of
# 67.4%. With the same metric we can honestly say "my system scores X%" rather
# than "it seems to work".
#
# THE DEFINITION (as theirs, with our clarifications):
#   a frame counts if the target is
#     a) in the CENTRAL THIRD of the frame on both axes, AND
#     b) within an acceptable DISTANCE range.
#   The result is the share of counted frames over the whole run.
#
# HONEST DIFFERENCES from the original (so as not to pass wishes off as facts):
#   * they zero out the remaining frames after a COLLISION -- we do not detect
#     collisions, so our figure is MORE OPTIMISTIC than theirs;
#   * they measure real distance, we use box size (a proxy);
#   * their run is a full office scenario, ours is an arbitrary recording.
# So the number is only ROUGHLY comparable, and that has to be stated plainly.

# Andon Labs Drone-Bench human baselines (for context in the report).
HUMAN_BASELINE = {
    "Reconstruct": 0.822,
    "Localize": 0.840,
    "Navigate": 0.920,
    "Detect": 0.721,
    "Follow": 0.674,
}


def in_center_third(cx, cy, frame_w, frame_h):
    """Whether the target centre is in the central third on BOTH axes.
    The central third by width is [W/3, 2W/3], i.e. |cx - W/2| < W/6."""
    return abs(cx - frame_w / 2) < frame_w / 6 and abs(cy - frame_h / 2) < frame_h / 6


def in_distance_band(distance_m, target_m, tolerance_m):
    """Whether the target is at an acceptable DISTANCE (in metres).

    This check used to work on box AREA, while the controller was already
    steering in metres. The metric and the control measured DIFFERENT quantities
    -- so the "within distance" line did not reflect what the system was aiming
    for. Both now take the setpoint from one source (the controller config)."""
    if distance_m is None or not target_m:
        return False
    return abs(distance_m - target_m) <= tolerance_m


class FollowScore:
    """Accumulates the Follow metric across the frames of a run."""

    def __init__(self, target_m=1.6, tolerance_m=0.6):
        # App passes the setpoint in from the controller config -- so the metric
        # measures exactly what the system controls.
        self._target_m = target_m
        self._tolerance_m = tolerance_m
        self._distances = []
        self.frames = 0          # every frame of the run
        self.centered = 0        # target in the central third
        self.in_range = 0        # target at an acceptable distance
        self.good = 0            # both at once (this is the Follow score)
        self.no_target = 0       # frames with no target at all

    def add(self, center, distance_m, frame_w, frame_h):
        """center=None means the target was not being tracked in this frame (zero points).

        distance_m is the READY-MADE distance, the same one the controller acts on
        (ControlError.distance_m). The metric does NOT compute it itself, and that
        matters: it once measured distance its own way, and the "within distance"
        line read 20.7% where the true figure was 37.1%. Tuning was being done
        against a ruler that measured something other than what the system
        controls.

        STALE FRAMES never reach here at all -- App filters them out. Otherwise
        the metric would mix "we cannot track" with "the video did not update":
        measured live, the stale frames alone accounted for 66.8% of "no target"
        on one flight. (That figure is not reproducible from the recordings --
        see perception/frame_quality.py -- but the reason for the filter does not
        depend on its exact size.)"""
        self.frames += 1
        if center is None:
            self.no_target += 1
            return
        if distance_m is not None:
            self._distances.append(distance_m)
        c = in_center_third(center[0], center[1], frame_w, frame_h)
        d = in_distance_band(distance_m, self._target_m, self._tolerance_m)
        self.centered += c
        self.in_range += d
        self.good += (c and d)

    @property
    def score(self):
        """The headline number: the share of frames with the target in the target zone."""
        return self.good / self.frames if self.frames else 0.0

    def report(self):
        """Report lines -- for the console and for the README alike."""
        n = self.frames or 1
        return [
            f"FOLLOW SCORE: {100 * self.score:.1f}%   "
            f"(Andon Labs human baseline — {100 * HUMAN_BASELINE['Follow']:.1f}%)",
            f"  within the central third: {100 * self.centered / n:.1f}%",
            f"  within target distance  : {100 * self.in_range / n:.1f}%",
            f"  no target visible       : {100 * self.no_target / n:.1f}%",
            f"  (distance band: {self._target_m:.2f} ± {self._tolerance_m:.2f} m)",
        ] + self._distance_lines()

    def _distance_lines(self):
        """Distance in metres from box height. Remember the MEASUREMENT LIMIT: the
        lower end of the range runs into K/frame_height (~1.33 m for 478 px) and
        means "no further than", not an exact reading."""
        if not self._distances:
            return []
        d = sorted(self._distances)
        return [
            f"  target distance: median {d[len(d) // 2]:.2f} m  "
            f"(range {d[0]:.2f}–{d[-1]:.2f} m)",
        ]
