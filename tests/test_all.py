# The project-wide regression suite. No dependencies (plain asserts), run with:
#   .venv/bin/python tests/test_all.py
#
# Purpose: after any change, confirm quickly that nothing broke -- from the
# value objects up to the full App pipeline and replay.

import contextlib
import io
import time
import os
import sys

import numpy as np

# So that `avis` imports regardless of where this was started from (project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- a tiny test harness (instead of pytest, to avoid the dependency) ---
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


@contextlib.contextmanager
def quiet():
    """Silence internal print() calls ([flight]/[rec]) for the duration."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


from avis.models import Command, ControlError, Detection


def person(x, tid, name="person", cls=0):
    """A 'person' Detection in the frame (for tracking tests)."""
    return Detection(id=tid, xyxy=(float(x), 100.0, float(x + 60), 300.0),
                     conf=0.9, cls=cls, name=name)


def err(ex=0.0, ey=0.0, area=30000.0, w=960, h=720):
    return ControlError(error_x=ex, error_y=ey, area=area,
                        norm_x=ex / (w / 2), norm_y=ey / (h / 2))


# ─────────────────────────────────────────────────────────────────────────
def test_models():
    print("\n[ models / Detection ]")
    d = Detection(id=1, xyxy=(10, 20, 30, 60), conf=0.5, cls=0)
    check("center is correct", d.center == (20.0, 40.0))
    check("area is correct", d.area == 20 * 40)
    check("name defaults to ''", d.name == "")
    c = Command(1, 2, 3)
    check("Command has 3 axes (yaw/vertical/forward)",
          (c.yaw, c.vertical, c.forward) == (1, 2, 3))


def test_error_calculator():
    print("\n[ ErrorCalculator ]")
    from avis.control import ErrorCalculator
    ec = ErrorCalculator()
    e = ec.compute(cx=640, cy=360, area=5000, frame_width=960, frame_height=720)
    check("error_x = cx - w/2", approx(e.error_x, 160))
    check("norm_x = ex/(w/2)", approx(e.norm_x, 160 / 480))
    check("area passes through unchanged", e.area == 5000)
    check("centred -> error 0", ErrorCalculator().compute(480, 360, 1, 960, 720).error_x == 0)


def test_controller():
    print("\n[ PID (the regulator itself) ]")
    from avis.control.controller import PID
    p = PID(kp=0.2, ki=0, kd=0)
    o1 = p.update(100, 0.1)
    check("PID: constant error -> stable output", approx(p.update(100, 0.1), o1))
    p.reset()
    check("PID.reset() clears the state", p._prev_error is None and p._integral == 0)
    check("PID clamps its output", abs(PID(kp=100, ki=0, kd=0).update(999, 0.1)) <= 100)
    # Following behaviour is now covered in tests/test_follow.py --
    # each axis has its own tests there.



def test_kalman():
    print("\n[ KalmanFilter ]")
    from avis.perception.tracking_filter import KalmanFilter
    kf = KalmanFilter()
    check("initialized is False at first", not kf.initialized)
    check("predict() with no state -> None", kf.predict(0.1) is None)
    x, y = kf.correct(100, 200)
    check("first correct -> the measurement unchanged", (x, y) == (100, 200))
    check("initialized is True after correct", kf.initialized)

    kf2 = KalmanFilter()
    kf2.correct(100, 240)
    kf2.predict(0.1); kf2.correct(120, 240)
    kf2.predict(0.1); kf2.correct(140, 240)
    px, _ = kf2.predict(0.1)
    check("predict extrapolates along the velocity (x grows)", px > 140)

    steps = []
    prev = px
    for _ in range(60):
        nx, _ = kf2.predict(0.1)
        steps.append(nx - prev); prev = nx
    check("blind decay: a later step is smaller than an early one",
          abs(steps[-1]) < abs(steps[0]))
    kf2.reset()
    check("reset() -> initialized False", not kf2.initialized)


def test_selector():
    print("\n[ TargetSelector: occlusion / reacquisition / click ]")
    from avis.perception.target_selector import TargetSelector

    s = TargetSelector()
    d = person(100, 7)
    got = s.select_at(85, 200, [d])           # 15px past the left edge of the box
    check("select_at forgives an imprecise hit", got is not None and got.id == 7)
    check("select_at far away -> None", TargetSelector().select_at(500, 450, [d]) is None)

    s2 = TargetSelector(); s2.select_target(person(100, 1))
    t, lost = s2.resolve([person(100, 1)])
    check("resolve by id returns the target", t is not None and not lost)

    # OCCLUSION with debounce: the first frames of a shrinking box are NOT yet
    # treated as a loss (that is most often detector flicker), only a sustained one.
    s3 = TargetSelector(occlusion_debounce=3)
    s3.select_target(person(100, 1))               # the reference area is large
    small = Detection(id=1, xyxy=(100, 100, 110, 130), conf=.9, cls=0)  # tiny
    t1, lost1 = s3.resolve([small])
    check("occlusion, frame 1: we still trust the box", t1 is not None and not lost1)
    s3.resolve([small])
    t3, lost3 = s3.resolve([small])
    check("occlusion, 3rd frame in a row -> (None, lost=True)", t3 is None and lost3)

    # a single flicker must NOT break tracking
    s3b = TargetSelector(occlusion_debounce=3)
    s3b.select_target(person(100, 1))
    s3b.resolve([small])                            # flickered
    t_ok, lost_ok = s3b.resolve([person(100, 1)])   # a normal box again
    check("flickered and recovered -> tracking intact", t_ok is not None and not lost_ok)

    s4 = TargetSelector(); s4.select_target(person(100, 1))
    t, lost = s4.resolve([person(105, 9)], predicted_center=(135, 200))
    check("reacquire: a new id next to the prediction", t is not None and s4.target_id == 9)

    s5 = TargetSelector(lost_grace_seconds=1.0); s5.select_target(person(100, 1))
    for _ in range(40):
        t, lost = s5.resolve([], predicted_center=(130, 200), dt=0.1)
    check("after the grace period -> (None, False) + cleared",
          t is None and not lost and s5.target_id is None)


def test_supervisor():
    print("\n[ FlightSupervisor ]")
    from avis.control.flight import FlightSupervisor, GROUNDED, HOVER, AUTO
    from avis.control.drone import NullDrone, SimDrone
    DT = 1 / 30
    C = Command(80, 60, 90)

    s = FlightSupervisor(NullDrone(), settle_after_takeoff=1.0)
    check("GROUNDED: update -> None", s.update(C, True, DT) is None)
    with quiet():
        s.takeoff()
    check("takeoff -> HOVER", s.state == HOVER)
    with quiet():
        s.arm()
    check("arm -> AUTO", s.state == AUTO)
    early = s.update(C, True, DT)
    check("settling: commands are 0 right after takeoff",
          early.yaw == 0 and early.forward == 0)
    with quiet():
        for _ in range(int(1.2 / DT)):
            late = s.update(C, True, DT)
    check("after settling the commands flow", late.yaw != 0)

    s2 = FlightSupervisor(NullDrone())
    with quiet():
        s2.takeoff(); s2.arm()
        for _ in range(int(10 / DT)):
            s2.update(None, False, DT)
    check("10 s with no target before acquisition -> does NOT land", s2.state == AUTO)

    s3 = FlightSupervisor(NullDrone(), settle_after_takeoff=0.5)
    with quiet():
        s3.takeoff(); s3.arm()
        for _ in range(int(0.6 / DT)):
            s3.update(Command(10, 0, 0), True, DT)
        for _ in range(int(9 / DT)):
            s3.update(None, False, DT)
    check("acquired, then 9 s of loss -> GROUNDED", s3.state == GROUNDED)

    class LowBat(NullDrone):
        def battery(self):
            return 5
    s4 = FlightSupervisor(LowBat())
    with quiet():
        s4.takeoff(); s4.arm()
        for _ in range(int(2.5 / DT)):
            s4.update(Command(10, 0, 0), True, DT)
    check("battery < 15% -> forced landing", s4.state == GROUNDED)

    s5 = FlightSupervisor(SimDrone(height=190), start_armed=True)
    check("altitude OFF: vertical passes through", s5._apply_altitude(40) == 40)
    s6 = FlightSupervisor(SimDrone(height=190), start_armed=True, altitude_limits=True)
    check("altitude ON: above the ceiling, up = 0", s6._apply_altitude(40) == 0)
    s7 = FlightSupervisor(SimDrone(height=40), start_armed=True, altitude_limits=True)
    check("altitude ON: below the floor, down = 0", s7._apply_altitude(-40) == 0)

    s8 = FlightSupervisor(NullDrone())
    check("_shape: large -> ceiling", s8._shape(500, 70) == 70)
    check("_shape: weak but non-zero -> min_speed", s8._shape(3, 70) == s8._min_speed)
    check("_shape: 0 -> 0", s8._shape(0, 70) == 0)

    # THREE separate ceilings. The numbers are NOT pinned here -- they are tuning
    # knobs and have already changed three times (55->80->50 for forward), each
    # time failing this test on correct code. What we pin is the INTENT: the
    # forward axis is the riskiest (the drone flies AT the operator), so its
    # ceiling must stay below the yaw ceiling.
    s9 = FlightSupervisor(NullDrone(), enable_forward=True)
    check("forward is the most cautious axis: ceiling below yaw/vertical",
          s9._max_forward < s9._max_speed)
    # And retreat is more cautious still than approach: the drone is blind behind.
    check("backward is more cautious than forward", s9._max_backward < s9._max_forward)
    # The forward ceiling is limited by the CAMERA, not the motors: past ~55 a
    # sharp pitch smears the frame and detection drops threefold (see flight.py).
    check("the forward ceiling stays out of the zone where detection is lost",
          s9._max_forward <= 55)
    # The ceilings must stay within the Tello protocol: rc accepts -100..100.
    check("the ceilings stay within the Tello limit (100)",
          s9._max_forward <= 100 and s9._max_speed <= 100
          and s9._max_backward <= 100)
    # And the sensitivity floor cannot beat the ceiling -- otherwise a command
    # would come out LARGER than allowed (_shape closes that trap with
    # min(floor, ceiling)).
    check("the sensitivity floor is below the ceilings",
          s9._min_forward < s9._max_forward and s9._min_speed < s9._max_speed)

    # We compare against the supervisor's OWN ceilings, not the numbers 55/70.
    # It used to be the other way: every tuning bump of a ceiling broke these
    # three tests even though clamping worked correctly. A test must check the
    # INVARIANT ("the command is clamped exactly at the ceiling"), not the
    # current value of a knob.
    lim = s9._limit(Command(yaw=999, vertical=999, forward=999))
    check("_limit: forward is clamped at the ceiling", lim.forward == s9._max_forward)
    check("_limit: yaw is clamped at the ceiling", lim.yaw == s9._max_speed)
    check("_limit: vertical is clamped at the ceiling", lim.vertical == s9._max_speed)

    # the battery cache used by the log
    s10 = FlightSupervisor(NullDrone())
    check("battery is None before takeoff", s10.battery is None)
    with quiet():
        s10.takeoff()
    check("battery is filled in on takeoff", s10.battery == 100)


def test_drone():
    print("\n[ Drone (the contract) ]")
    from avis.control.drone import Drone, NullDrone, SimDrone, TelloDrone
    for cls in (NullDrone, SimDrone, TelloDrone):
        check(f"{cls.__name__} implements Drone", issubclass(cls, Drone))
    check("SimDrone.height() is settable", SimDrone(height=123).height() == 123)
    check("NullDrone.battery() = 100", NullDrone().battery() == 100)

    sent = {}
    class FakeTello:
        def send_rc_control(self, lr, fb, ud, yaw): sent.update(lr=lr, fb=fb, ud=ud, yaw=yaw)
    TelloDrone(FakeTello()).send(Command(yaw=5, vertical=6, forward=7))
    check("TelloDrone: channels map correctly (left_right=0)",
          sent["lr"] == 0 and sent["fb"] == 7 and sent["ud"] == 6 and sent["yaw"] == 5)


def test_bugfixes():
    print("\n[ Fixed bugs (regression guards) ]")
    from avis.perception.target_selector import TargetSelector
    from avis.control.follow import FollowController
    from avis.control.flight import FlightSupervisor, GROUNDED, HOVER, AUTO, BUSY
    from avis.control.drone import Drone, NullDrone

    # BUG #1: in PRED the supervisor must see has_target=False and stop motion
    class Spy(Drone):
        def __init__(s): s.sent = []
        def takeoff(s): pass
        def land(s): s.sent.append("LAND")
        def emergency(s): pass
        def send(s, c): s.sent.append((round(c.yaw), round(c.vertical), round(c.forward)))
        def battery(s): return 90
        def height(s): return 100

    DT = 1 / 10
    d = Spy(); sup = FlightSupervisor(d, enable_forward=True, settle_after_takeoff=0)
    with quiet():
        sup.takeoff()
        while sup.state == BUSY:
            pass
        sup.arm()
    d.sent.clear()
    with quiet():
        for _ in range(int(3 / DT)):
            sup.update(Command(100, 0, 100), False, DT)   # THERE IS NO TARGET
    nz = [x for x in d.sent if isinstance(x, tuple) and any(x)]
    check("#1 no target -> it stops (does not chase a ghost)", len(nz) <= 6)

    # BUG #2: the target class is NOT overwritten, even if YOLO changes the class
    s = TargetSelector()
    s.select_target(person(100, 1))                        # a PERSON was selected (cls=0)
    chair_same_id = Detection(id=1, xyxy=(100, 100, 160, 300), conf=.9, cls=56, name="chair")
    s.resolve([chair_same_id])                             # YOLO suddenly says "chair"
    check("#2 the target class stayed a person", s._target_cls == 0)

    # BUG #3 is now covered in tests/test_follow.py (test_controller: PRED)

    # BUG #6: a click between two boxes takes the one whose CENTRE is closer
    s2 = TargetSelector()
    big = Detection(id=1, xyxy=(100, 100, 300, 500), conf=.9, cls=0, name="person")
    small = Detection(id=2, xyxy=(180, 380, 240, 440), conf=.9, cls=56, name="chair")
    picked = s2.select_at(200, 200, [big, small])          # the click is closer to the PERSON's centre
    check("#6 the click takes the nearest by centre, not the smallest", picked.id == 1)

    # BUG #9: the reacquisition radius was reduced
    check("#9 reacquisition radius ceiling <= 250px",
          TargetSelector()._reacquire_radius_max <= 250)

    # BUG #12: commands are rounded, not truncated
    from avis.control.drone import TelloDrone
    sent = {}
    class FakeTello:
        def send_rc_control(self, lr, fb, ud, yaw): sent.update(fb=fb, ud=ud, yaw=yaw)
    TelloDrone(FakeTello()).send(Command(yaw=12.9, vertical=-12.9, forward=12.4))
    check("#12 round(), not int()",
          sent["yaw"] == 13 and sent["ud"] == -13 and sent["fb"] == 12)


def test_arm_and_backward():
    print("\n[ G during takeoff + slow retreat ]")
    import time as _t
    from avis.control.drone import Drone, NullDrone
    from avis.control.flight import FlightSupervisor, AUTO, HOVER, BUSY

    # G pressed DURING takeoff must not be dropped
    class SlowDrone(Drone):
        def takeoff(s): _t.sleep(1.0)
        def land(s): pass
        def emergency(s): pass
        def send(s, c): pass
        def battery(s): return 90
        def height(s): return 100

    sup = FlightSupervisor(SlowDrone())
    with quiet():
        sup.takeoff()
        sup.toggle()                       # G while still taking off
    check("G during takeoff: the state is still BUSY", sup.state == BUSY)
    _t.sleep(1.4)
    check("after takeoff tracking armed BY ITSELF", sup.state == AUTO)

    # a second G must disarm (not arm again)
    with quiet():
        sup.toggle()
    check("a second G disarms tracking", sup.state == HOVER)

    # retreat is twice as slow as approach
    s2 = FlightSupervisor(NullDrone(), enable_forward=True)
    fwd = s2._limit(Command(0, 0, 999)).forward
    bwd = s2._limit(Command(0, 0, -999)).forward
    check("forward is clamped at the ceiling", fwd == s2._max_forward)
    # The point is not the number 25 but TWO requirements at once: retreat is
    # more cautious than approach, yet not below the sensitivity floor --
    # otherwise the backward axis is dead.
    check("backward: below forward, but EFFECTIVE",
          bwd == -s2._max_backward
          and s2._max_backward < s2._max_forward
          and s2._max_backward >= s2._min_forward)


def test_key_debounce():
    print("\n[ Key debounce (one press = one action) ]")
    from avis.app import App
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor, AUTO, HOVER
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator
    from avis.control.follow import FollowController
    from avis.view import HeadlessRenderer

    sup = FlightSupervisor(NullDrone())
    app = App(source=None, detector=None, selector=TargetSelector(),
              tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
              controller=FollowController(), renderer=HeadlessRenderer(),
              supervisor=sup)
    with quiet():
        sup.takeoff()
        while sup.state == "BUSY":
            pass
        # TWO identical events in a row (duplicate from the window + terminal)
        app._handle_key(ord('g'))
        app._handle_key(ord('g'))
    check("a duplicated G does not cancel itself -- it stays AUTO", sup.state == AUTO)


def test_nonblocking_takeoff():
    print("\n[ Takeoff does not block the loop (the video does not freeze) ]")
    import time as _t
    from avis.control.drone import Drone
    from avis.control.flight import FlightSupervisor, BUSY, HOVER

    class SlowDrone(Drone):
        """Imitates the Tello: takeoff replies only after 2 s (really up to 20)."""
        def takeoff(s): _t.sleep(2.0)
        def land(s): pass
        def emergency(s): pass
        def send(s, c): pass
        def battery(s): return 90
        def height(s): return 100

    sup = FlightSupervisor(SlowDrone())
    t0 = _t.time()
    with quiet():
        sup.takeoff()
    elapsed = _t.time() - t0
    check(f"takeoff() returns control immediately ({elapsed:.2f}s)", elapsed < 0.3)
    check("the state is BUSY during takeoff", sup.state == BUSY)
    check("nothing is sent to the motors during takeoff",
          sup.update(Command(50, 0, 0), True, 0.1) is None)
    _t.sleep(2.3)
    check("once finished the state is HOVER", sup.state == HOVER)


def test_key_handling():
    print("\n[ Keys: codes with high bits (macOS/Qt) ]")
    from avis.app import App
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor, GROUNDED, HOVER, AUTO
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator, FollowController
    from avis.view.renderer import Renderer

    class Head(Renderer):
        def set_mouse_callback(s, c): pass
        def draw(s, *a, **k): pass
        def is_open(s): return True

    def make():
        sup = FlightSupervisor(NullDrone())
        app = App(source=None, detector=None, selector=TargetSelector(),
                  tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
                  controller=FollowController(), renderer=Head(), supervisor=sup)
        return app, sup

    HIGH = 0x100000        # the high bits macOS/Qt adds

    # plain ASCII works
    app, sup = make()
    with quiet():
        app._handle_key(ord('t'))
    check("plain code 't' -> takeoff", sup.state == HOVER)
    with quiet():
        app._handle_key(ord('g'))
    check("plain code 'g' -> tracking", sup.state == AUTO)

    # THE REGRESSION we catch: a code with high bits must work too
    app2, sup2 = make()
    with quiet():
        app2._handle_key(HIGH | ord('t'))
    check("code 't' WITH HIGH BITS -> takeoff", sup2.state == HOVER)
    with quiet():
        app2._handle_key(HIGH | ord('g'))
    check("code 'g' WITH HIGH BITS -> tracking", sup2.state == AUTO)
    with quiet():
        app2._handle_key(HIGH | ord('l'))
    check("code 'l' WITH HIGH BITS -> landing", sup2.state == GROUNDED)

    # quitting must be caught in both forms as well
    app3, _ = make()
    check("quit: plain Esc", app3._handle_key(27) is True)
    check("quit: 'q' with high bits", app3._handle_key(HIGH | ord('q')) is True)


def test_integration_live():
    print("\n[ Integration: the full App (live stub) ]")
    from avis.app import App
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator, FollowController
    from avis.view.renderer import Renderer

    class LiveSource:
        """The frames MUST differ from each other. This used to be np.zeros(),
        i.e. byte-identical frames, which no real camera ever produces (even a
        still scene carries sensor noise). The pipeline would now rightly treat
        them as stale and skip the detector."""
        def __init__(s, n=6): s._n = n
        def read(s):
            if s._n <= 0: return None
            s._n -= 1
            frame = np.zeros((480, 640, 3), np.uint8)
            frame[:] = s._n + 1          # every frame differs, as with a real camera
            return frame
        def release(s): pass

    class StubDet:
        def __init__(s): s.c = 0
        def detect(s, f):
            s.c += 1
            return [Detection(id=1, xyxy=(450, 220, 510, 340), conf=.9, cls=0)]

    class Head(Renderer):
        def set_mouse_callback(s, c): pass
        def draw(s, *a, **k): pass
        def is_open(s): return True

    det = StubDet()
    app = App(source=LiveSource(), detector=det, selector=TargetSelector(),
              tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
              controller=FollowController(), renderer=Head(),
              supervisor=FlightSupervisor(NullDrone(), start_armed=True, enable_forward=True))
    app._pipeline.select_target(Detection(id=1, xyxy=(450, 220, 510, 340), conf=.9, cls=0))
    with quiet():
        app.run()
    check("App.run() without crashes", True)
    check("the detector was called on every frame", det.c == 6)





def test_integration_replay():
    print("\n[ Integration: record -> replay ]")
    from avis.app import App
    from avis.replay import FlightRecorder, ReplaySource
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator, FollowController
    from avis.view.renderer import Renderer

    name = "_pytest_tmp"
    with quiet():
        rec = FlightRecorder(name)
        x = 200
        for i in range(20):
            x += 12
            frame = np.zeros((480, 640, 3), np.uint8)
            dets = [Detection(id=1, xyxy=(x, 220, x + 50, 300), conf=.9, cls=0, name="person")]
            # The baseline is written too: what reached the motors, state, battery.
            rec.write(frame, 1 / 30, dets, 1,
                      sent=Command(yaw=5, vertical=-2, forward=30),
                      state="AUTO", battery=100)
        rec.close()

    rs = ReplaySource(name)
    f = rs.read()
    check("ReplaySource.read() returns a frame", f is not None and f.shape == (480, 640, 3))
    check("replayed_dt is preserved", approx(rs.replayed_dt(), 1 / 30, 1e-3))
    check("replayed_detections restores Detection", len(rs.replayed_detections()) == 1)
    base = rs.baseline()
    check("baseline: that flight's commands are preserved",
          base["sent"] is not None and base["sent"]["forward"] == 30.0)
    check("baseline: state and battery are preserved",
          base["state"] == "AUTO" and base["battery"] == 100)
    rs.release()

    class Head(Renderer):
        def set_mouse_callback(s, c): pass
        def draw(s, *a, **k): pass
        def is_open(s): return True

    with quiet():
        app = App(source=ReplaySource(name), detector=None, selector=TargetSelector(),
                  tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
                  controller=FollowController(), renderer=Head(),
                  supervisor=FlightSupervisor(NullDrone(), start_armed=True, enable_forward=True))
        app.run()
    check("App ran through in replay mode", app._replaying)

    for ext in (".mp4", ".jsonl"):
        p = os.path.join("recordings", name + ext)
        if os.path.exists(p):
            os.remove(p)
    check("temporary replay files cleaned up", True)


def test_arm_messages():
    """The message must distinguish "takeoff in progress" from "there was no takeoff".

    A real case: both states printed "still taking off", and the operator waited
    120 seconds for a takeoff nobody had started -- T had not registered, and the
    program kept confirming that all was well. A log that hides the problem is
    worse than silence."""
    print("\n[ Takeoff state messages ]")
    from avis.control.flight import AUTO, GROUNDED, HOVER, FlightSupervisor
    from avis.control.drone import NullDrone

    def say(fn):
        """Capture what the supervisor printed."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        return buf.getvalue()

    # 1. The drone is on the ground, there was NO takeoff.
    s = FlightSupervisor(NullDrone())
    out = say(s.arm)
    check("on the ground: it says to press T", "T" in out)
    check("on the ground: it does NOT claim a takeoff is in progress",
          "still taking off" not in out)
    check("on the ground: the intent is remembered anyway", s._arm_requested)
    check("on the ground: the state did not change", s.state == GROUNDED)

    # 2. A takeoff IS actually running (we imitate the background operation).
    s2 = FlightSupervisor(NullDrone())
    s2._busy = True
    out2 = say(s2.arm)
    check("during takeoff: it says the takeoff is in progress", "still taking off" in out2)
    check("during takeoff: the intent is remembered", s2._arm_requested)

    # 3. Airborne -- the ordinary arming of tracking.
    s3 = FlightSupervisor(NullDrone())
    with quiet():
        s3.takeoff()
        for _ in range(200):                      # wait for the background thread
            if s3.state == HOVER:
                break
            time.sleep(0.01)
    out3 = say(s3.arm)
    check("airborne: tracking arms immediately", s3.state == AUTO)
    check("airborne: it says AUTO", "AUTO" in out3)


def test_settle_counts_in_hover():
    """Post-takeoff settling must run ON THE CLOCK, not from the moment G is pressed.

    The bug: the countdown only ticked in the AUTO branch, while HOVER returned
    earlier. However long the operator hovered picking a target, after pressing G
    the drone executed no command for another 5 seconds. Measured on two flights:
    exactly 5.1 s from entering AUTO to the first command, and on one of them the
    whole AUTO episode lasted just that -- the operator concluded tracking was
    broken and switched it off."""
    print("\n[ Post-takeoff settling ]")
    from avis.control.flight import AUTO, HOVER, FlightSupervisor
    from avis.control.drone import NullDrone
    DT = 1 / 30
    C = Command(60, 0, 40)

    # Hovered for 6 seconds, then armed tracking -- commands must flow AT ONCE.
    s = FlightSupervisor(NullDrone(), settle_after_takeoff=5.0)
    with quiet():
        s.takeoff()
        for _ in range(int(6 / DT)):          # 6 s in HOVER, picking a target
            s.update(C, True, DT)
        check("no commands flow in HOVER", s.state == HOVER)
        s.arm()
        first = s.update(C, True, DT)
    check("after 6 s of hovering, G gives a command AT ONCE",
          first is not None and first.yaw != 0)

    # And arming right after takeoff -- settling must engage.
    s2 = FlightSupervisor(NullDrone(), settle_after_takeoff=5.0)
    with quiet():
        s2.takeoff(); s2.arm()
        early = s2.update(C, True, DT)
    check("right after takeoff the commands are held back", early.yaw == 0)
    with quiet():
        for _ in range(int(5.2 / DT)):
            late = s2.update(C, True, DT)
    check("5 s after takeoff the commands flow", late.yaw != 0)
    check("the state stayed AUTO", s2.state == AUTO)


def test_stale_frames():
    """A stale frame is the previous frame shown a second time.

    More dangerous than a broken one: it looks perfect, so the system
    confidently "sees" the target in a picture that is seconds old. Measured
    live in flight: 24% of frames were repeats, the longest run 97 frames
    (5.1 s). Those figures cannot be reproduced from a recording -- the .mp4 is
    re-encoded, so consecutive frames are no longer byte-identical and
    is_stale() sees almost nothing on replay (see frame_quality.py)."""
    print("\n[ Stale frames ]")
    import cv2
    from avis.perception.frame_quality import FrameQuality
    from avis.pipeline import PRED, VIS, FrameResult, TrackingPipeline

    rng = np.random.default_rng(11)

    def scene():
        base = rng.integers(0, 255, size=(30, 40, 3), dtype=np.uint8)
        return cv2.GaussianBlur(np.repeat(np.repeat(base, 16, 0), 16, 1), (31, 31), 0)

    q = FrameQuality()
    a, b = scene(), scene()
    check("the first frame is not stale", not q.is_stale(a))
    check("a new frame is not stale", not q.is_stale(b))
    check("THE SAME frame is stale", q.is_stale(b))
    check("a repeat of a repeat is stale too", q.is_stale(b))
    check("after a repeat, a new frame is clean", not q.is_stale(a))
    check("None does not break the check", FrameQuality().is_stale(None) is False)

    # --- The key point: a stale frame does NOT count as "target visible" ---
    r = FrameResult(status=VIS, frame_stale=True)
    check("stale VIS: has_target = False", not r.has_target)
    check("stale VIS: blind = True", r.blind)
    fresh = FrameResult(status=VIS)
    check("fresh VIS: has_target = True", fresh.has_target)
    check("fresh VIS: blind = False", not fresh.blind)

    # --- The pipeline must not run the detector on the same picture ---
    class CountingDetector:
        def __init__(self): self.calls = 0
        def detect(self, frame):
            self.calls += 1
            return []

    class Sel:
        target_id = None
        def resolve(self, det, pred=None, dt=0.0): return None, False
    class Trk:
        initialized = False
        def predict(self, dt): return None
        def reset(self): pass
    class Ctl:
        def update(self, e, dt): return None
        def reset(self): pass

    det = CountingDetector()
    pipe = TrackingPipeline(det, Sel(), Trk(), None, Ctl())
    frame = scene()
    pipe.process(frame, 0.05)          # the first one is fresh
    calls_after_first = det.calls
    res = pipe.process(frame, 0.05)    # the same frame object
    check("the detector is NOT called on a stale frame", det.calls == calls_after_first)
    check("the result is marked stale", res.frame_stale)
    check("a stale frame does not yield status VIS", res.status != VIS)

    # --- The supervisor does not count stale frames as a target loss ---
    from avis.control.flight import AUTO, FlightSupervisor
    from avis.control.drone import NullDrone
    DT = 1 / 30
    s = FlightSupervisor(NullDrone(), settle_after_takeoff=0.0)
    with quiet():
        s.takeoff(); s.arm()
        s.update(Command(10, 0, 0), True, DT)
        for _ in range(int(20 / DT)):
            s.update(None, False, DT, blind=True)
    check("20 s of frozen video -> the drone does not land", s.state == AUTO)


def test_metrics():
    """The metric must measure THE SAME thing the system controls.

    Bug history: the metric computed distance on its own while the control
    measured it differently -- and it reported 20.7% "at the right distance"
    instead of the real 37.1%. Distance now arrives READY-MADE, from the same
    formula.

    The setpoints here are given EXPLICITLY (target_m=...), because this tests
    the counting mechanics rather than the setpoint value: otherwise every shift
    of the FollowConfig default would break the arithmetic of these checks."""
    print("\n[ Quality metric (Follow score) ]")
    from avis.metrics import FollowScore, in_center_third, in_distance_band

    W, H = 648, 478
    check("the frame centre is in the central third", in_center_third(W / 2, H / 2, W, H))
    check("the frame edge is NOT in the central third", not in_center_third(10, H / 2, W, H))
    # The width boundary of the third is |cx - W/2| < W/6.
    check("just past the third -- not counted",
          not in_center_third(W / 2 + W / 6 + 1, H / 2, W, H))

    check("distance inside the band", in_distance_band(0.8, 0.75, 0.2))
    check("distance outside the band", not in_distance_band(1.5, 0.75, 0.2))
    check("an unknown distance does not count", not in_distance_band(None, 0.75, 0.2))

    fs = FollowScore(target_m=0.75, tolerance_m=0.2)
    fs.add((W / 2, H / 2), 0.75, W, H)          # centred and at distance
    fs.add((W / 2, H / 2), 2.00, W, H)          # centred but far
    fs.add((10, H / 2), 0.75, W, H)             # at distance but at the edge
    fs.add(None, None, W, H)                    # no target was followed
    check("the score counts full hits only", approx(fs.score, 0.25))
    check("centring is counted separately", fs.centered == 2)
    check("distance is counted separately", fs.in_range == 2)
    check("frames without a target are counted", fs.no_target == 1)

    # THE KEY POINT: the metric takes the READY-MADE distance and recomputes
    # nothing -- even when the value lies past the measurement limit (0.70 m
    # against a 1.33 m limit). Judging reliability is the job of whoever
    # measured, not of the metric.
    close = FollowScore(target_m=0.75, tolerance_m=0.2)
    close.add((W / 2, H / 2), 0.70, W, H)
    check("a close distance counts (not a blind zone)", close.score == 1.0)
    check("the real distance appears in the report",
          any("target distance: median" in s for s in close.report()))

    empty = FollowScore()
    check("an empty run does not divide by zero", empty.score == 0.0)
    check("an empty run does not print a distance",
          not any("target distance: median" in s for s in empty.report()))


def test_frame_quality():
    """A broken frame means 'I cannot see', NOT 'there is no target'.

    Replayed against flight_20260926_224656: detection succeeds on 49% of clean
    frames against 4% of broken ones, and during long target losses 24% of
    frames are broken against a 12% baseline. So the absence of a box on such a
    frame proves nothing, and the loss timer must stop."""
    print("\n[ Frame quality / packet loss ]")
    from avis.perception.frame_quality import FrameQuality

    rng = np.random.default_rng(7)

    def clean(h=240, w=320):
        """A natural image: smooth gradients with no seams on the 16px grid."""
        base = rng.integers(0, 255, size=(h // 16, w // 16, 3), dtype=np.uint8)
        big = np.repeat(np.repeat(base, 16, axis=0), 16, axis=1)
        # Blur it so the block boundaries DISAPPEAR -- that is how a real scene looks.
        import cv2
        return cv2.GaussianBlur(big, (31, 31), 0)

    def broken(h=240, w=320):
        """H.264 corruption as it ACTUALLY looks: some 16x16 blocks are taken
        from the wrong place (damaged motion vectors). The texture INSIDE a
        block survives -- what tears is the GRID. Flat blocks would be a
        dishonest test: their internal step is zero and any detector would catch
        them trivially."""
        img = clean(h, w)
        for y in range(0, h - 16, 16):
            for x in range(0, w - 16, 16):
                if rng.random() < 0.45:                     # ~45% of blocks displaced
                    sy = int(rng.integers(0, h - 16)) // 16 * 16
                    sx = int(rng.integers(0, w - 16)) // 16 * 16
                    img[y:y + 16, x:x + 16] = img[sy:sy + 16, sx:sx + 16]
        return img

    q = FrameQuality()
    for _ in range(40):                       # warm-up on clean frames
        q.is_corrupt(clean())
    check("a clean frame is not marked broken", not q.is_corrupt(clean()))
    check("a broken frame is detected", q.is_corrupt(broken()))

    # The adaptive-threshold trap: a long run of garbage must NOT become the norm.
    q2 = FrameQuality()
    for _ in range(40):
        q2.is_corrupt(clean())
    with_streak = [q2.is_corrupt(broken()) for _ in range(60)]
    check("a long run of garbage does not 'become the norm'", all(with_streak))
    check("after the run, a clean frame is clean again", not q2.is_corrupt(clean()))

    # Empty/invalid input must not break the pipeline.
    check("None does not break the detector", FrameQuality().is_corrupt(None) is False)
    check("a flat patch is not considered broken",
          FrameQuality().is_corrupt(np.zeros((120, 160, 3), np.uint8)) is False)

    # --- The key point: a broken frame does NOT accelerate target loss ---
    from avis.control.flight import FlightSupervisor, AUTO, GROUNDED
    from avis.control.drone import NullDrone
    DT = 1 / 30

    s = FlightSupervisor(NullDrone(), settle_after_takeoff=0.0)
    with quiet():
        s.takeoff(); s.arm()
        s.update(Command(10, 0, 0), True, DT)          # the target had been acquired
        for _ in range(int(20 / DT)):                  # 20 s of solid garbage
            s.update(None, False, DT, blind=True)
    check("20 s of broken frames -> the drone does NOT land", s.state == AUTO)

    # And a genuine loss (frames clean, no target) must still land it, as before.
    s2 = FlightSupervisor(NullDrone(), settle_after_takeoff=0.0)
    with quiet():
        s2.takeoff(); s2.arm()
        s2.update(Command(10, 0, 0), True, DT)
        for _ in range(int(9 / DT)):
            s2.update(None, False, DT, blind=False)
    check("a genuine 9 s loss -> lands (safety not broken)", s2.state == GROUNDED)

    # Garbage must not allow blind flight: nothing should reach the motors.
    s3 = FlightSupervisor(NullDrone(), settle_after_takeoff=0.0)
    with quiet():
        s3.takeoff(); s3.arm()
        s3.update(Command(50, 0, 50), True, DT)
        for _ in range(int(3 / DT)):
            out = s3.update(None, False, DT, blind=True)
    check("on garbage the drone holds still instead of flying blind",
          out is not None and out.forward == 0 and out.yaw == 0)

    # blind=False by default -- old call sites behave as before.
    s4 = FlightSupervisor(NullDrone(), settle_after_takeoff=0.0)
    with quiet():
        s4.takeoff(); s4.arm()
        s4.update(Command(10, 0, 0), True, DT)
        for _ in range(int(9 / DT)):
            s4.update(None, False, DT)
    check("compatibility: a call without blind behaves as before", s4.state == GROUNDED)


def main():
    print("=" * 60)
    print("AVIS REGRESSION SUITE")
    print("=" * 60)
    for t in (test_models, test_error_calculator, test_controller, test_kalman,
              test_selector, test_supervisor, test_drone, test_key_handling,
              test_bugfixes, test_nonblocking_takeoff,
              test_arm_and_backward, test_key_debounce,
              test_arm_messages, test_metrics, test_frame_quality,
              test_settle_counts_in_hover, test_stale_frames,
              test_integration_live, test_integration_replay):
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            _FAILURES.append(f"{t.__name__} RAISED: {type(e).__name__}: {e}")
            print(f"  ✗ EXCEPTION in {t.__name__}: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    if _FAILURES:
        print("Failures:")
        for f in _FAILURES:
            print(f"  - {f}")
    print("=" * 60)
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
