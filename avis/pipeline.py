# THE FRAME PROCESSING PIPELINE -- pure computation, no I/O.
#
# WHY IT WAS EXTRACTED: all of this logic used to live inside App.run() -- a
# 150-line method combining a dozen responsibilities: timing, detection, clicks,
# tracking, control, safety, statistics, recording, rendering and the keyboard.
# You had to read the whole thing to understand any part of it.
#
# The split is now clean:
#   TrackingPipeline -- "what we see and where to fly" (computation, no side effects)
#   App              -- "where frames come from, where to draw, what was pressed" (orchestration)
#
# The classes know no more about each other than they need to: the pipeline has
# no notion of windows, keys or recording, and App does not know how tracking
# works inside.

from dataclasses import dataclass, field

from avis.models import Command, ControlError

# Target-tracking states.
VIS = "VIS"     # target reliably visible -- we steer by measurement
PRED = "PRED"   # occluded or briefly gone -- we steer by the Kalman prediction
LOST = "LOST"   # gone for a long time -- there is no target


@dataclass
class FrameResult:
    """Everything the pipeline learned about a frame. An immutable snapshot for App."""
    detections: list = field(default_factory=list)
    status: str = LOST
    center: tuple = None          # smoothed or predicted target position
    predicted: tuple = None       # the Kalman hypothesis (for drawing the reticle)
    error: ControlError = None
    command: Command = None       # what the controller wants (not yet safety-filtered)
    target_id: int = None
    off_frame: bool = False       # the prediction left the frame
    detect_seconds: float = 0.0
    # The frame is broken by packet loss -- it shows garbage, not a scene. This is
    # NOT the same as "there is no target": detection on such frames collapses by
    # roughly an order of magnitude (measured 49% on clean frames against 4% on
    # broken ones), so the absence of a box here proves nothing.
    frame_corrupt: bool = False
    # The frame is STALE -- the decoder gave us nothing new and we are seeing the
    # previous one a second time. It looks flawless, and that is exactly why it
    # is more dangerous than a broken one: the system confidently "sees" the
    # target in a picture that is already seconds old.
    frame_stale: bool = False

    @property
    def has_target(self) -> bool:
        """True ONLY when the target is genuinely visible IN A FRESH frame.

        Two traps, both of which cost real flights:
          * in PRED the position exists, but it is a Kalman hypothesis. This
            used to read "center is not None", and the drone chased a ghost;
          * on a STALE frame the target is perfectly visible -- the frame is just
            old. Measured: 5.1 s of control from one and the same picture."""
        return self.status == VIS and not self.frame_stale

    @property
    def blind(self) -> bool:
        """The frame carries NO new information about the target at all.

        Two different causes, identical consequences: the frame fell apart
        (corruption) or the frame did not update (staleness). Whoever reads this
        has no right to treat the absence of a target as a loss -- absence of
        evidence is not evidence of absence."""
        if self.frame_stale:
            return True
        return self.frame_corrupt and self.status != VIS


class TrackingPipeline:
    """Frame -> detections -> target -> smoothing -> control command."""

    def __init__(self, detector, selector, tracker, error_calculator, controller,
                 on_event=None, quality=None):
        self._detector = detector
        self._selector = selector
        self._tracker = tracker
        self._error_calculator = error_calculator
        self._controller = controller
        # Frame quality control. Not an interface and not in the composition
        # root: there is one implementation, and the pipeline works without it
        # (quality=False disables it).
        if quality is None:
            from avis.perception.frame_quality import FrameQuality
            quality = FrameQuality()
        self._quality = quality or None
        # Where to report events (target change and so on). A function rather
        # than print -- so the pipeline does not depend on the output mechanism
        # and stays silent in tests.
        self._on_event = on_event or (lambda msg: None)

        self._last_box = None          # last target box (for the distance)
        self._last_area = 0.0
        self._active_target_id = None  # we watch for target substitution

    # -- Target selection (delegated to the selector) -------------------
    def select_at(self, x, y, detections):
        return self._selector.select_at(x, y, detections)

    def select_target(self, det):
        return self._selector.select_target(det)

    @property
    def target_id(self):
        return self._selector.target_id

    # -- The main step -------------------------------------------------
    def process(self, frame, dt, detections=None) -> FrameResult:
        """detections=None -> we detect ourselves; otherwise we take them ready (replay)."""
        import time
        h, w = frame.shape[:2]

        # PROCESSING A STALE FRAME IS POINTLESS -- and this is logic, not an
        # optimisation. On the same picture the detector gives the same answer we
        # already know: 33 ms spent for nothing. Worse, the result would look
        # fresh and would reach the motors. So we return immediately, giving the
        # Kalman filter the real dt -- time passes, the target moves, and the
        # prediction has to keep up.
        #
        # ON THE STALE-FRAME SHARE. Live measurements put it at roughly a quarter
        # of frames (and 47% of AUTO frames on one flight), i.e. a quarter of the
        # detector's work going nowhere. Those figures cannot be reproduced from
        # the recordings -- see the note in perception/frame_quality.py -- so
        # treat them as indicative rather than exact.
        if self._quality and self._quality.is_stale(frame):
            predicted, off_frame = self._predict(dt, w, h)
            return FrameResult(
                status=PRED if predicted is not None else LOST,
                center=predicted, predicted=predicted, off_frame=off_frame,
                target_id=self._selector.target_id, frame_stale=True)

        corrupt = self._quality.is_corrupt(frame) if self._quality else False

        t0 = time.time()
        if detections is None:
            detections = self._detector.detect(frame)
        detect_seconds = time.time() - t0

        predicted, off_frame = self._predict(dt, w, h)
        # TIME FOR THE PREDICTION AND TIME FOR THE LOSS are DIFFERENT things, and
        # here they diverge. The Kalman filter gets the real dt: time passes, the
        # target really is moving, and the prediction has to keep up. But the
        # loss countdown STOPS on a broken frame: we failed to see the target not
        # because it is gone but because the frame is garbage. Otherwise a couple
        # of seconds of Wi-Fi interference would force the drone to declare the
        # target lost and start hovering, or landing.
        target, lost = self._selector.resolve(detections, predicted,
                                              0.0 if corrupt else dt)

        result = FrameResult(detections=detections, predicted=predicted,
                             off_frame=off_frame, detect_seconds=detect_seconds,
                             target_id=self._selector.target_id,
                             frame_corrupt=corrupt)

        if target is not None:
            result.status = VIS
            result.center = self._on_visible(target)
        elif lost:
            # NOTE: nothing here bounds how long PRED may last. The Kalman
            # velocity decay means the prediction freezes rather than running
            # away, but it freezes indefinitely: measured PRED runs of 87, 77, 63
            # and 56 frames (4-6 s) on flight_20261002_185914. has_target is
            # False throughout, so the supervisor hovers correctly -- but if you
            # want the loop to reach its "land on loss" decision on time, this is
            # where a cap belongs.
            result.status = PRED
            result.center = predicted
        else:
            result.status = LOST
            self._reset_tracking()

        if result.center is not None:
            cx, cy = result.center
            # We pass box ONLY in the VIS state: in PRED the box is stale and the
            # distance must not be measured from it. The error then arrives with
            # distance_m=None, and the distance axis decides for itself what to
            # do -- it holds the last trustworthy measurement briefly, then stops.
            result.error = self._error_calculator.compute(
                cx, cy, self._last_area, w, h,
                box=self._last_box if result.status == VIS else None)
            result.command = self._controller.update(result.error, dt)

        return result

    # -- Internal steps ------------------------------------------------
    def _predict(self, dt, w, h):
        """The Kalman prediction, clamped into the frame bounds.

        The clamp is needed because a constant-velocity model extrapolates
        OUTWARDS once the target has left the frame. Without it the controller
        would chase a point off screen, and re-acquisition would look for the
        target where boxes never appear."""
        if not self._tracker.initialized:
            return None, False
        predicted = self._tracker.predict(dt)
        if predicted is None:
            return None, False
        px, py = predicted
        cpx = min(max(px, 0.0), w - 1.0)
        cpy = min(max(py, 0.0), h - 1.0)
        return (cpx, cpy), (cpx != px or cpy != py)

    def _on_visible(self, target):
        """Target visible: catch substitution, correct the filter, remember the box."""
        if self._active_target_id is not None and target.id != self._active_target_id:
            # Re-acquisition onto a DIFFERENT object: the position jumps by
            # hundreds of pixels. Without a reset the PID derivative spikes and
            # the Kalman filter reads the jump as an insane velocity -- hence the
            # sharp jerks.
            #
            # HOW OFTEN THIS FIRES, because it is more often than it looks:
            # ByteTrack reissues ids 39 times across one 218-second flight
            # (flight_20261002_185914, 25 distinct ids). So both the filter and
            # the controller are wiped 39 times per flight.
            self._on_event(f"target changed #{self._active_target_id} -> "
                           f"#{target.id} ({target.name}) -- resetting filter and controller")
            self._tracker.reset()
            self._controller.reset()
        self._active_target_id = target.id
        self._last_area = target.area
        self._last_box = target.xyxy
        return self._tracker.correct(*target.center)

    def _reset_tracking(self):
        self._tracker.reset()
        self._controller.reset()
        self._active_target_id = None
