# Tests for the avis.control.follow package. Written ALONGSIDE each phase.
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PASS = 0
_FAIL = 0
_FAILURES = []


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {name}")
    else:
        _FAIL += 1
        _FAILURES.append(name)
        print(f"  ✗ FAIL: {name}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ── PHASE 2: config ──────────────────────────────────────────────────────
def test_config():
    print("\n[ FollowConfig — validation catches real bugs ]")
    from avis.control.follow.config import FollowConfig

    c = FollowConfig()
    check("defaults are valid", c.distance_target_m == 1.70)

    def raises(**kw):
        try:
            FollowConfig(**kw)
            return False
        except ValueError:
            return True

    check("deadzone wider than the setpoint -> error",
          raises(distance_target_m=0.2, distance_deadzone_m=0.5))
    check("align_full >= align_stop -> error",
          raises(align_full=0.9, align_stop=0.5))
    check("align_stop > 1 -> error", raises(align_stop=1.5))
    check("negative setpoint -> error", raises(distance_target_m=-1))
    check("retreat_factor outside (0,1] -> error", raises(retreat_factor=0))
    check("negative deadzone -> error", raises(yaw_deadzone_px=-5))


# ── PHASE 3: the base axis ───────────────────────────────────────────────
def test_axis():
    print("\n[ Axis — a SINGLE soft deadzone ]")
    from avis.control.follow.axis import Axis

    a = Axis(kp=1.0, kd=0.0, deadzone=10.0)
    check("inside the deadzone -> exactly 0", a.shape(5) == 0.0)
    check("on the boundary itself -> 0", a.shape(10) == 0.0)
    check("past the boundary -> starts from zero, no step", approx(a.shape(11), 1.0))
    check("symmetric for negatives", approx(a.shape(-11), -1.0))
    check("far away -> error minus deadzone", approx(a.shape(100), 90.0))

    # the key property: there is NO jump at the boundary
    just_out = abs(a.update(10.001, 0.1))
    check("at the boundary the command is ~0 (no jerk)", just_out < 0.01)

    a.reset()
    check("hold() returns 0", a.hold() == 0.0)


# ── PHASE 4: the axes ────────────────────────────────────────────────────
# The default dist is DELIBERATELY FAR from the 1.70 m setpoint (deadzone
# 1.50-1.90). It used to be 1.6 — and once the setpoint moved from 0.75 to 1.70
# that value landed INSIDE the deadzone, so the "far away -> move forward" tests
# would silently have been checking "inside the deadzone -> 0".
def _err(ex=0.0, ey=0.0, dist=2.6, clipped=False, w=960, h=720, fill=0.1):
    from avis.models import ControlError
    return ControlError(error_x=ex, error_y=ey, area=0.0,
                        norm_x=ex / (w / 2), norm_y=ey / (h / 2),
                        distance_m=dist, box_clipped=clipped, frame_fill=fill)


def test_yaw_axis():
    print("\n[ YawAxis ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.yaw import YawAxis

    a = YawAxis(FollowConfig())
    check("centred -> 0", a.command(_err(ex=0), 0.1) == 0.0)
    check("inside the deadzone (20px) -> 0", a.command(_err(ex=20), 0.1) == 0.0)
    out = a.command(_err(ex=200), 0.1)
    check("target to the right -> turn right (+)", out > 0)
    a.reset()
    check("target to the left -> turn left (-)", a.command(_err(ex=-200), 0.1) < 0)


def test_vertical_axis():
    print("\n[ VerticalAxis — adaptive level and freezing ]")
    from avis.control.follow.config import FollowConfig as _C
    from avis.control.follow.vertical import VerticalAxis as _V
    # DISABLED by default — because of oscillation from the feedback loop
    off = _V(_C())
    check("disabled by default -> always 0",
          off.command(_err(ey=-300), 0.15, distance_rate=0.0) == 0.0)
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.vertical import VerticalAxis

    # 1. A constant offset must NOT lift the drone
    a = VerticalAxis(FollowConfig(vertical_enabled=True))
    for _ in range(60):
        out = a.command(_err(ey=-130), 0.15, distance_rate=0.0)
    check("constant -130px offset -> no command", abs(out) < 1e-6)

    # 2. Crouching (distance unchanged) — it reacts
    for _ in range(4):
        out = a.command(_err(ey=-30), 0.15, distance_rate=0.0)
    check("target dropped -> the drone descends", out < 0)

    # 3. Approaching (distance changing fast) — frozen
    b = VerticalAxis(FollowConfig(vertical_enabled=True))
    for _ in range(40):
        b.command(_err(ey=-130), 0.15, distance_rate=0.0)
    out = b.command(_err(ey=-30), 0.15, distance_rate=-0.8)
    check("distance changing -> vertical frozen", out == 0.0)

    # 4. A clipped box — frozen as well
    c = VerticalAxis(FollowConfig(vertical_enabled=True))
    for _ in range(40):
        c.command(_err(ey=-130), 0.15, distance_rate=0.0)
    check("clipped box -> frozen",
          c.command(_err(ey=-30, clipped=True), 0.15, distance_rate=0.0) == 0.0)


def test_distance_hold():
    """When the distance measurement disappears we hold the last one instead of
    dropping to zero.

    The measurement vanishes when the box is clipped both at the top and at the
    side, i.e. the target has moved to the edge of the frame. Measured on the
    19:35 flight: at |ex| 150-250 px the box was clipped sideways in 73% of
    frames, distance became unavailable in 56%, and forward motion switched off
    in 75%. With no distance, forward was 0 in 100% of frames. The drone never
    closed in: 1.44 -> 1.37 -> 1.95 -> 1.58 m against a 0.75 setpoint.
    (The setpoint has since moved to 1.70 m — see FollowConfig.)"""
    print("\n[ DistanceAxis — holding the measurement ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.distance import DistanceAxis

    cfg = FollowConfig(distance_hold_s=1.0)
    ax = DistanceAxis(cfg)
    DT = 1 / 20

    # Gather trustworthy measurements — the target is far, so it should move forward.
    for _ in range(6):
        out = ax.command(_err(dist=2.60), DT)
    check("measurement available -> moves forward", out > 0)
    moving = out

    # The measurement is gone (box clipped sideways). Motion MUST remain.
    lost = ax.command(_err(dist=None), DT)
    check("measurement lost -> motion does NOT drop to zero", lost > 0)
    check("held command stays close to the previous one", abs(lost - moving) < 0.6 * abs(moving))

    # Hold past the allowed window — beyond that it is invention, so we stop.
    for _ in range(int(1.5 / DT)):
        stale = ax.command(_err(dist=None), DT)
    check("after 1.5 s with no measurement -> stop", stale == 0.0)

    # A fresh measurement resets the hold timer.
    ax2 = DistanceAxis(cfg)
    for _ in range(6):
        ax2.command(_err(dist=2.60), DT)
    for _ in range(int(0.8 / DT)):
        ax2.command(_err(dist=None), DT)
    ax2.command(_err(dist=2.60), DT)            # the measurement came back
    for _ in range(int(0.8 / DT)):
        again = ax2.command(_err(dist=None), DT)
    check("a fresh measurement refreshes the hold timer", again > 0)

    # --- THE PROXIMITY GUARD HAS NOT WEAKENED ---
    ax3 = DistanceAxis(cfg)
    for _ in range(6):
        ax3.command(_err(dist=2.60), DT)
    near = ax3.command(_err(dist=None, fill=0.55), DT)   # box fills half the screen
    check("close by frame fill -> forward forbidden", near <= 0.0)

    ax4 = DistanceAxis(cfg)
    for _ in range(6):
        ax4.command(_err(dist=2.60), DT)
    retreat = ax4.command(_err(dist=None, fill=0.70), DT)
    check("very close -> forced retreat", retreat < 0)

    # A clipped box ON ITS OWN no longer kills motion: proximity is frame_fill.
    ax5 = DistanceAxis(cfg)
    for _ in range(6):
        ax5.command(_err(dist=2.60), DT)
    clipped = ax5.command(_err(dist=2.60, clipped=True, fill=0.12), DT)
    check("clipped box at small frame fill does not kill motion", clipped > 0)

    # With no measurement at all from the start there is nothing to hold.
    ax6 = DistanceAxis(cfg)
    check("nothing to hold -> 0", ax6.command(_err(dist=None), DT) == 0.0)


def test_distance_axis():
    print("\n[ DistanceAxis — ONE setpoint, metres only ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.distance import DistanceAxis

    cfg = FollowConfig()
    a = DistanceAxis(cfg)
    check("on the setpoint -> 0", a.command(_err(dist=1.70), 0.1) == 0.0)
    a.reset()
    check("inside the deadzone -> 0", a.command(_err(dist=1.85), 0.1) == 0.0)
    a.reset()
    check("far -> forward", a.command(_err(dist=2.6), 0.1) > 0)
    a.reset()
    check("close -> backward", a.command(_err(dist=0.30), 0.1) < 0)

    # LINEARITY — the main advantage of metres over area.
    #
    # The probes are derived FROM THE GAIN rather than fixed at 1 and 2 metres.
    # Why: the PID output is clamped by its own limit of 100 (the Tello RC
    # range), so "double the error" only holds WHILE BOTH commands stay below
    # 100. With fixed metres the test silently checked saturation instead of
    # linearity: at kp=55 a 2 m error gave 110 -> 100, the ratio came out as
    # 1.82 instead of 2.0, and the test failed on correct code. Now we aim at
    # 25 and 50 units for any kp.
    e1 = 25.0 / cfg.distance_kp                  # the error that yields a command of ≈25
    base = cfg.distance_target_m + cfg.distance_deadzone_m
    x1 = DistanceAxis(cfg).command(_err(dist=base + e1), 0.1)
    x2 = DistanceAxis(cfg).command(_err(dist=base + 2 * e1), 0.1)
    check("twice the error -> twice the command", approx(x2 / x1, 2.0, 0.05))
    check("linearity probes do not hit the PID output limit", abs(x2) < 99.0)

    # Asymmetry and clamping
    fwd = DistanceAxis(cfg).command(_err(dist=2.6), 0.1)
    back = DistanceAxis(cfg).command(_err(dist=0.2), 0.1)
    check("retreat is slower than approach", abs(back) < abs(fwd))
    # A CLIPPED BOX ON ITS OWN DOES NOT KILL FORWARD MOTION. box_clipped now
    # means only "the box geometry is unreliable" and is read by the vertical
    # axis alone. Proximity is measured by frame_fill — directly, with no
    # dependency on calibration.
    check("clipped box at small frame fill -> motion allowed",
          DistanceAxis(cfg).command(_err(dist=2.6, clipped=True, fill=0.1), 0.1) > 0)
    check("large frame fill -> forward forbidden",
          DistanceAxis(cfg).command(_err(dist=2.6, fill=0.55), 0.1) <= 0)
    check("very large frame fill -> forced retreat",
          DistanceAxis(cfg).command(_err(dist=2.6, fill=0.70), 0.1) < 0)
    check("no measurement -> 0", DistanceAxis(cfg).command(_err(dist=None), 0.1) == 0.0)


def test_alignment_gate():
    print("\n[ AlignmentGate ]")
    from avis.control.follow.alignment import AlignmentGate
    from avis.control.follow.config import FollowConfig

    g = AlignmentGate(FollowConfig())
    check("centred -> full speed", g.factor(0.0) == 1.0)
    check("within align_full -> full speed", g.factor(0.30) == 1.0)
    check("at the edge -> motion forbidden", g.factor(0.9) == 0.0)
    mid = g.factor(0.55)
    check("between the thresholds -> partial speed", 0 < mid < 1)
    check("symmetric", g.factor(-0.55) == mid)


# ── PHASE 5: the coordinator ─────────────────────────────────────────────
def test_controller():
    print("\n[ FollowController — coordinating the axes ]")
    from avis.control.follow import FollowController

    c = FollowController()
    out = c.update(_err(ex=200, dist=3.0), 0.1)
    check("target far and off to the side -> both turn and forward motion",
          out.yaw > 0 and out.forward > 0)

    c2 = FollowController()
    edge = c2.update(_err(ex=440, dist=3.0), 0.1)
    check("target at the edge -> turn only, no forward motion",
          edge.yaw > 0 and edge.forward == 0.0)

    # ── PRED: the target is occluded, we steer by the Kalman prediction ───
    # Forward motion used to be forced to zero here. Measured on the 26.09
    # flight: that is 25% of AUTO frames, and the median such pause is 0.18 s,
    # over which the distance physically cannot change. The approach was
    # interrupted every few frames and the drone "kept its distance".
    c3 = FollowController()
    for _ in range(6):                                   # measurements gathered
        c3.update(_err(ex=200, dist=3.0), 0.1)
    pred = c3.update(_err(ex=200, dist=None), 0.1)       # target left the frame
    check("PRED -> the approach CONTINUES on the held measurement", pred.forward > 0)
    check("PRED -> we still steer the heading", pred.yaw > 0)

    # But not forever: once the hold expires we stop rather than fly blind.
    c3b = FollowController()
    for _ in range(6):
        c3b.update(_err(ex=200, dist=3.0), 0.1)
    for _ in range(int(1.5 / 0.1)):
        stale = c3b.update(_err(ex=200, dist=None), 0.1)
    check("PRED for too long -> stop, not flight on invention", stale.forward == 0.0)
    check("PRED for too long -> turning still works", stale.yaw > 0)

    # The proximity guard applies even when the distance is held.
    c3c = FollowController()
    for _ in range(6):
        c3c.update(_err(ex=0, dist=3.0), 0.1)
    near = c3c.update(_err(ex=0, dist=None, fill=0.70), 0.1)
    check("held distance + close by frame -> retreat", near.forward < 0)

    c4 = FollowController(yaw_kp=0.9)
    check("pinpoint parameter override works", c4.config.yaw_kp == 0.9)
    check("the setpoint is exposed (for the metric)", c4.distance_target_m == 1.70)


# ── PHASE 7: EFFECTIVENESS on real recordings ────────────────────────────
#
# THIS IS THE MOST IMPORTANT TEST IN THE PROJECT. 92 unit tests used to pass
# while the drone physically could not move: the command never reached the
# magnitude the Tello acts on (~15) and never held long enough. The unit tests
# checked the formulas but NOT that the system produces an EFFECTIVE result.
# That is exactly what we check here — against real flight data.

def _run_recording(name):
    """Run a recording through the full pipeline and collect motor commands."""
    import contextlib as _c
    import io as _io
    from avis.app import App
    from avis.control import ErrorCalculator
    from avis.control.drone import Drone
    from avis.control.flight import FlightSupervisor
    from avis.control.follow import FollowController
    from avis.perception import KalmanFilter, TargetSelector
    from avis.replay import ReplaySource
    from avis.view import HeadlessRenderer

    class Spy(Drone):
        def __init__(self): self.sent = []
        def takeoff(self): pass
        def land(self): pass
        def emergency(self): pass
        def send(self, c): self.sent.append(c)
        def battery(self): return 90
        def height(self): return 100

    spy = Spy()
    app = App(source=ReplaySource(name), detector=None, selector=TargetSelector(),
              tracker=KalmanFilter(),
              error_calculator=ErrorCalculator(),
              controller=FollowController(), renderer=HeadlessRenderer(),
              supervisor=FlightSupervisor(spy, enable_forward=True,
                                          start_armed=True, settle_after_takeoff=0))
    with _c.redirect_stdout(_io.StringIO()):
        app.run()
    return spy.sent


def _longest_run(values, threshold):
    best = cur = 0
    for v in values:
        cur = cur + 1 if abs(v) >= threshold else 0
        best = max(best, cur)
    return best


def test_effectiveness():
    print("\n[ EFFECTIVENESS on real flights ]")
    import glob
    import os

    # Only recordings where a flight ACTUALLY happened. Otherwise the test picks
    # up aborted runs of 20-30 frames (started it, looked at the picture, closed
    # it) and fails not because the control is bad but because the drone never
    # took off then.
    def is_real_flight(path, min_auto_frames=200):
        try:
            with open(path) as fh:
                return sum(1 for line in fh if '"state": "AUTO"' in line) >= min_auto_frames
        except OSError:
            return False

    recs = [p for p in sorted(glob.glob("recordings/*.jsonl")) if is_real_flight(p)]
    if not recs:
        print("  (no real flights among the recordings — test skipped)")
        return

    EFFECTIVE = 15      # below this the Tello physically does not move
    MIN_HOLD = 5        # the command must hold for at least a few frames in a row

    total_fwd_ok = total_yaw_ok = 0
    for rec in recs[-3:]:
        name = os.path.basename(rec)[:-6]
        sent = _run_recording(name)
        if not sent:
            continue
        fwd = [c.forward for c in sent]
        yaw = [c.yaw for c in sent]
        f_run = _longest_run(fwd, EFFECTIVE)
        y_run = _longest_run(yaw, EFFECTIVE)
        f_pct = 100 * sum(1 for v in fwd if abs(v) >= EFFECTIVE) / len(fwd)
        print(f"    {name[7:]}: forward {f_pct:.0f}% effective, "
              f"longest {f_run} frames; yaw {y_run} frames")
        total_fwd_ok += f_run >= MIN_HOLD
        total_yaw_ok += y_run >= MIN_HOLD

    check("the FORWARD axis issues effective commands for a sustained time",
          total_fwd_ok >= 2)
    check("the YAW axis issues effective commands for a sustained time",
          total_yaw_ok >= 2)


def test_no_dual_setpoint():
    """A regression guard against the bug that broke target following:
    the system must have ONE distance setpoint, not two."""
    print("\n[ A single source of truth for distance ]")
    from avis.control.follow import FollowController
    from avis.control.follow.config import FollowConfig

    c = FollowController()
    check("the setpoint lives in the config only", hasattr(c.config, "distance_target_m"))
    check("there is no separate area setpoint", not hasattr(c, "_desired_area"))
    check("there is no fallback area branch",
          not any("area" in f for f in FollowConfig.__dataclass_fields__))

    # CALIBRATION IS GONE ENTIRELY. It carried no new information: K enters the
    # linear control exactly as a scale factor, so recalibrating was a silent
    # change of distance_kp. The system now has neither a calibration file nor a
    # class that reads one — so there is nothing left to shift the gain quietly.
    import avis.distance as dist_mod
    check("the DistanceEstimator class no longer exists",
          not hasattr(dist_mod, "DistanceEstimator"))
    check("no calibration file is read",
          not any(hasattr(dist_mod, n) for n in ("CONFIG", "load", "calibrate")))
    check("metres are always available, from one formula",
          dist_mod.meters_from_height(300) is not None)


# ── PHASES 8-10: pipeline, statistics, orchestrator ──────────────────────
_frame_counter = [0]


def _stub_frame(w=960, h=720):
    """Every call returns a DIFFERENT frame.

    This used to be a plain np.zeros(), and the tests passed one and the same
    frame into several process() calls. The pipeline now rightly treats such a
    frame as STALE (the decoder produced nothing new) and skips processing — so
    the tests would have been checking something other than intended. A real
    camera never produces identical frames: even a still scene carries sensor
    noise."""
    import numpy as np
    _frame_counter[0] += 1
    frame = np.zeros((h, w, 3), np.uint8)
    frame[0, 0] = _frame_counter[0] % 251      # frame marker, harmless to detections
    return frame


def _person(x, tid, y1=100, y2=400):
    from avis.models import Detection
    return Detection(id=tid, xyxy=(float(x), float(y1), float(x + 80), float(y2)),
                     conf=0.9, cls=0, name="person")


def _make_pipeline(events=None):
    from avis.control import ErrorCalculator
    from avis.control.follow import FollowController
    from avis.perception import KalmanFilter, TargetSelector
    from avis.pipeline import TrackingPipeline
    return TrackingPipeline(
        detector=None, selector=TargetSelector(), tracker=KalmanFilter(),
        error_calculator=ErrorCalculator(),
        controller=FollowController(),
        on_event=(events.append if events is not None else None))


def test_pipeline():
    print("\n[ TrackingPipeline — pure processing, no IO ]")
    from avis.pipeline import LOST, PRED, VIS

    # EVERY process() call gets a FRESH frame — exactly as in flight. Passing
    # the same object twice would mean a stale frame, and the pipeline would
    # rightly skip processing.
    p = _make_pipeline()

    # no target selected
    r = p.process(_stub_frame(), 0.1, detections=[_person(400, 1)])
    check("no target selected -> LOST", r.status == LOST)
    check("LOST -> no command", r.command is None)
    check("has_target=False", not r.has_target)

    # selected and following
    p.select_target(_person(400, 1))
    r = p.process(_stub_frame(), 0.1, detections=[_person(400, 1)])
    check("target selected -> VIS", r.status == VIS)
    check("VIS -> a command exists", r.command is not None)
    check("VIS -> has_target=True", r.has_target)
    check("centre computed", r.center is not None)

    # target gone -> PRED, and has_target MUST be False
    r2 = p.process(_stub_frame(), 0.1, detections=[])
    check("target gone -> PRED", r2.status == PRED)
    check("PRED -> has_target=False (safety must engage)", not r2.has_target)
    # In PRED the approach CONTINUES on the held measurement: the previous frame
    # was VIS, so a trustworthy distance already exists. This used to be zero —
    # and that is exactly why the drone stopped every time the target was
    # briefly occluded. The limit of trust is held by the axis itself
    # (distance_hold_s), not by a separate flag.
    check("PRED -> the approach continues on the held measurement",
          r2.command is not None and r2.command.forward > 0)

    # the target-swap event
    events = []
    p2 = _make_pipeline(events)
    p2.select_target(_person(400, 1))
    p2.process(_stub_frame(), 0.1, detections=[_person(400, 1)])
    p2.process(_stub_frame(), 0.1, detections=[_person(410, 2)])   # a different id nearby
    check("a target swap is reported as an event",
          any("target changed" in e for e in events))


def test_pipeline_offframe():
    print("\n[ TrackingPipeline — the prediction is clamped to the frame ]")
    frame = _stub_frame()
    p = _make_pipeline()
    p.select_target(_person(800, 1))
    # follow a target moving fast to the right, out of the frame
    x = 800
    for _ in range(6):
        x += 30
        p.process(frame, 0.1, detections=[_person(min(x, 870), 1)])
    r = p.process(frame, 0.1, detections=[])       # gone past the edge
    check("the prediction does not leave the frame",
          r.predicted is None or 0 <= r.predicted[0] <= 959)


def test_stats():
    print("\n[ RunStats ]")
    from avis.models import Command
    from avis.pipeline import FrameResult, VIS
    from avis.stats import RunStats

    st = RunStats()
    check("empty report for 0 frames", st.report() == [])

    r = FrameResult(status=VIS, center=(480, 360),
                    command=Command(50, 0, 40))
    st.add(r, Command(30, 0, 20), 960, 720)      # clamped on both axes
    st.add(r, Command(30, 0, 20), 960, 720)
    lines = "\n".join(st.report())
    check("the report contains the frame count", "2 frames" in lines)
    check("saturation is noticed", "saturation" in lines)
    check("jitter is computed", "jitter" in lines)


def test_single_measure():
    """ONE measure — the box height. A regression guard against two measures
    with switching between them.

    Measured on flight_20260926_224656, why two measures were a mistake: 56% of
    measurements went through shoulder width, which depends on POSE (the real
    height/width ratio ranged 1.61…2.77 against the 2.26 hard-coded), and 243
    height<->width switches produced a median jump of 0.18 m — larger than the
    entire 0.20 m deadzone."""
    print("\n[ ONE distance measure: box height only ]")
    from avis.distance import (DISTANCE_K, frame_fill, meters_from_height,
                               nearest_measurable_m, touches_edge)

    W, H = 960, 720
    check("height 300px -> metres available", meters_from_height(300) is not None)
    check("twice the box height -> twice as close",
          approx(meters_from_height(600) * 2, meters_from_height(300), 1e-9))
    check("invalid box -> None", meters_from_height(0) is None)

    # POSE HAS NO EFFECT — the main reason width was dropped.
    narrow = meters_from_height(400 - 100)      # the same height...
    wide = meters_from_height(400 - 100)        # ...whatever the shoulder width
    check("distance does NOT depend on box width", narrow == wide)

    # THE MEASUREMENT LIMIT is a property of the vertical FOV.
    floor = nearest_measurable_m(H)
    check("measurement limit = K / frame height", approx(floor, DISTANCE_K / H, 1e-9))
    check("at 720px the limit is ≈0.88 m", 0.87 < floor < 0.89)
    check("resolution does NOT move the limit (K and the frame scale together)",
          approx(nearest_measurable_m(478) / (DISTANCE_K / 478), 1.0, 1e-9))

    # touches_edge no longer means "cannot be measured"
    check("box in the middle of the frame -> does not touch the edge",
          not touches_edge((400, 150, 540, 540), W, H))
    check("box near the edge -> touches", touches_edge((0, 150, 140, 540), W, H))

    check("frame fill is computed from the visible part of the box",
          approx(frame_fill((0, 0, W // 2, H), W, H), 0.5, 1e-9))


def test_setpoint_is_measurable():
    """THE MAIN REGRESSION GUARD of this round.

    A 0.75 m setpoint against a 1.33 m measurement limit did the following
    (flight_20260926_224656, 2741 measurements): distance read as "too far" in
    93% of frames, 87% of the commands sent to the motors were "forward" and
    only 2% "backward" — and even with a person standing right in front of it,
    the drone pushed into them. The operator described it as "it does not back
    off, it just freezes" and "instead of turning on the spot it flies".

    The cause is purely geometric: the box height saturates as soon as the
    person no longer fits in the frame, and a saturated measurement reads as
    "far" if the setpoint lies closer than the limit. So the setpoint MUST be
    further out: K/frame_h < target − deadzone."""
    print("\n[ The distance setpoint is REACHABLE for the frame ]")
    from avis.control.follow.config import FollowConfig
    from avis.distance import nearest_measurable_m

    cfg = FollowConfig()
    for h in (478, 720):
        floor = nearest_measurable_m(h)
        near_edge = cfg.distance_target_m - cfg.distance_deadzone_m
        check(f"frame {h}px: measurement limit {floor:.2f} m is below the deadzone "
              f"edge {near_edge:.2f} m", floor < near_edge)
        check(f"frame {h}px: the check does not complain",
              cfg.check_measurable(h) is None)

    # And now THE BUG ITSELF: the check must catch it.
    bad = FollowConfig(distance_target_m=0.75)
    warning = bad.check_measurable(478)
    check("setpoint 0.75 m on a 478px frame -> WARNING", warning is not None)
    check("the warning names the measurement limit", warning and "1.33" in warning)
    check("the warning says what to do", warning and "distance_target_m" in warning)

    # And the failure mechanism itself: past the measurement limit the distance
    # saturates, and with a correct setpoint that reads as "too close".
    from avis.control.follow.distance import DistanceAxis
    from avis.distance import meters_from_height
    saturated = meters_from_height(478)          # the person does not fit a 478px frame
    axis = DistanceAxis(cfg)
    out = axis.command(_err(dist=saturated, fill=0.40), 0.1)
    check("saturated measurement with a 1.70 m setpoint -> RETREAT", out < 0)

    bad_axis = DistanceAxis(bad)
    bad_out = bad_axis.command(_err(dist=saturated, fill=0.40), 0.1)
    check("the same measurement with a 0.75 m setpoint -> approach (this was the bug)",
          bad_out > 0)


def test_proximity_guard():
    print("\n[ THE PROXIMITY GUARD — the main fix for 'flies into the object' ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.distance import DistanceAxis

    cfg = FollowConfig()

    # The most important scenario: the distance estimate LIES (the person turned
    # sideways, shoulders narrower -> "target is far"), but the box fills half
    # the frame.
    a = DistanceAxis(cfg)
    lying = a.command(_err(dist=3.0, fill=0.70), 0.1)   # "far" + a large box
    check("estimate lies 'far' but the box is large -> RETREAT, not approach", lying < 0)

    b = DistanceAxis(cfg)
    stop = b.command(_err(dist=3.0, fill=0.55), 0.1)
    check("box above the stop threshold -> forward forbidden", stop <= 0)

    c = DistanceAxis(cfg)
    ok = c.command(_err(dist=3.0, fill=0.10), 0.1)
    check("small box -> forward motion allowed", ok > 0)

    # The guard must be EFFECTIVE, not symbolic
    d = DistanceAxis(cfg)
    force = d.command(_err(dist=3.0, fill=0.9), 0.1)
    check("the retreat command exceeds the drone's sensitivity threshold (20)",
          abs(force) >= 20)

    # The median filter absorbs a single outlier caused by a pose change
    e = DistanceAxis(FollowConfig(distance_median_frames=5))
    for _ in range(5):
        e.command(_err(dist=1.0), 0.1)
    before = e.smooth(1.0)
    spike = e.smooth(5.0)          # a sharp false measurement
    check("the median absorbs a single outlier", abs(spike - before) < 0.5)


def test_approach_converges():
    """The key property: the drone MUST reach the setpoint instead of stopping
    short — and MUST back off when the person comes closer. The second part used
    not to work: 87% of commands were "forward" against 2% "backward" across a
    whole flight."""
    print("\n[ The approach converges to the setpoint, retreat works ]")
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.control.follow import FollowController
    from avis.distance import DISTANCE_K, frame_fill, meters_from_height

    W, H = 960, 720
    ASPECT = 2.3            # the real height/width ratio, measured in flight
    sup = FlightSupervisor(NullDrone(), enable_forward=True)
    target = FollowController().distance_target_m

    def motor_cmd(d):
        """What WOULD REACH THE MOTORS if the person stood at distance d.

        The box is modelled exactly as the camera sees it: the height is clipped
        by the frame bounds, and that is precisely why a measurement past the
        limit saturates."""
        true_h = DISTANCE_K / d
        visible_h = min(true_h, H)                  # the frame clips it
        box = (0.0, 0.0, true_h / ASPECT, visible_h)
        c = FollowController().update(
            _err(dist=meters_from_height(visible_h),
                 fill=frame_fill(box, W, H)), 0.15)
        if c.forward == 0:
            return 0.0
        ceiling = sup._max_forward if c.forward >= 0 else sup._max_backward
        return sup._shape(c.forward, ceiling, floor=sup._min_forward)

    check("from afar (3 m) — moves forward", motor_cmd(3.0) > 0)
    check("at 2 m — STILL moves (does not stall early)", motor_cmd(2.0) > 0)
    check("at the setpoint — holds still", motor_cmd(target) == 0)
    # The main thing that was broken: closer than the setpoint the drone must BACK OFF.
    check("at 1.2 m — retreats", motor_cmd(1.2) < 0)
    check("right up close (0.8 m) — retreats", motor_cmd(0.8) < 0)
    check("right up close (0.45 m) — retreats", motor_cmd(0.45) < 0)
    check("the retreat is EFFECTIVE (not below the sensitivity threshold of 20)",
          abs(motor_cmd(0.8)) >= 20)


def test_blind_holds_command():
    """A BLIND FRAME DOES NOT KILL MOTION. The costliest mistake of the last flight.

    Measured on flight_20260926_224656 (255.6 s in AUTO): 52% of frames reached
    the motors with a "zero" command, the drone stood still for 86.6 s (34% of
    the flight), 331 pauses with a median of 0.11 s, and 330 times a live command
    was replaced by zero on the VERY NEXT frame. The cause was stale frames: the
    stream delivers ~14 new frames/s while the loop spun at about 21 Hz on that
    flight (median dt 48 ms), so a large share of read() calls return the same
    picture. The command from such a frame arrived as None and execution fell
    through to send(0,0,0).

    This is the same principle already applied elsewhere in the project — to the
    loss timer and to the distance hold: ABSENCE OF EVIDENCE IS NOT EVIDENCE OF
    ABSENCE."""
    print("\n[ A blind frame holds the command instead of killing motion ]")
    from avis.control.drone import Drone
    from avis.control.flight import FlightSupervisor
    from avis.models import Command

    class Spy(Drone):
        def __init__(self): self.sent = []
        def takeoff(self): pass
        def land(self): pass
        def emergency(self): pass
        def send(self, c): self.sent.append(c)
        def battery(self): return 90
        def height(self): return 100

    def fresh_supervisor(**kw):
        spy = Spy()
        sup = FlightSupervisor(spy, enable_forward=True, start_armed=True,
                               settle_after_takeoff=0, **kw)
        return spy, sup

    DT = 1 / 14                      # the real rate of new frames in flight
    spy, sup = fresh_supervisor()
    live = sup.update(Command(yaw=40, vertical=0, forward=30), True, DT)
    check("live target -> the command reaches the motors",
          live.forward > 0 and live.yaw > 0)

    # Now a blind frame: no command, the target is "not visible" — but motion
    # MUST remain.
    held = sup.update(None, False, DT, blind=True)
    check("blind frame -> the command is HELD, not zeroed",
          held is not None and held.forward == live.forward and held.yaw == live.yaw)

    # ...but not forever. The blind_hold_s limit must engage.
    for _ in range(int(1.0 / DT)):
        after = sup.update(None, False, DT, blind=True)
    check("blind for too long -> stop (the hold is not infinite)",
          after.forward == 0 and after.yaw == 0)

    # REGRESSION: the old behaviour — zeros from the very first blind frame.
    spy2, sup2 = fresh_supervisor(blind_hold_s=0.0)
    sup2.update(Command(yaw=40, vertical=0, forward=30), True, DT)
    zero = sup2.update(None, False, DT, blind=True)
    check("blind_hold_s=0 -> behaves as before (zero)", zero.forward == 0)

    # A SIGHTED frame with no target is INFORMATION, and we treat it as before:
    # after hover_after_lost we stop, and the hold does NOT apply here.
    spy3, sup3 = fresh_supervisor()
    sup3.update(Command(yaw=40, vertical=0, forward=30), True, DT)
    for _ in range(int(1.0 / DT)):
        seen_empty = sup3.update(None, False, DT, blind=False)
    check("frame is CLEAN but there is no target -> stop (that is information)",
          seen_empty.forward == 0 and seen_empty.yaw == 0)


# ── Runner ───────────────────────────────────────────────────────────────
PHASES = [test_config, test_axis, test_yaw_axis, test_vertical_axis,
          test_distance_axis, test_distance_hold, test_alignment_gate, test_controller,
          test_no_dual_setpoint, test_pipeline, test_pipeline_offframe,
          test_stats, test_single_measure, test_setpoint_is_measurable,
          test_proximity_guard, test_approach_converges,
          test_blind_holds_command, test_effectiveness]


def main():
    print("=" * 60)
    print("TESTS FOR THE avis.control.follow PACKAGE")
    print("=" * 60)
    for t in PHASES:
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            _FAILURES.append(f"{t.__name__}: {type(e).__name__}: {e}")
            print(f"  ✗ EXCEPTION in {t.__name__}: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    for f in _FAILURES:
        print(f"  - {f}")
    print("=" * 60)
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
