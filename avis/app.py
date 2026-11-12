# App -- the ORCHESTRATOR. It does exactly three things:
#   1. spins the frame loop and decides when to stop;
#   2. connects I/O (camera, window, keys, recording) to the pipeline;
#   3. passes the pipeline's command through the safety filter to the motors.
#
# All the COMPUTATIONAL logic lives elsewhere:
#   TrackingPipeline -- detection, tracking, control (avis/pipeline.py)
#   FlightSupervisor -- flight safety (avis/control/flight.py)
#   RunStats, FollowScore -- metrics (avis/stats.py, avis/metrics.py)
#
# All of this used to live inside run() -- a 150-line method you had to read in
# full to understand any part of it. Now App does not know HOW tracking works,
# and the pipeline knows nothing about windows or keys.

import time

import cv2  # needed only for window input constants (keys, mouse)

from avis.control.flight import GROUNDED, FlightSupervisor
from avis.metrics import FollowScore
from avis.perception import FrameSource
from avis.pipeline import VIS, TrackingPipeline
from avis.stats import RunStats
from avis.view import Renderer
from avis.view.keys import TerminalKeys

# Quit keys. Esc (27) is reliable in any layout and with CapsLock on.
# Letters support both Latin and Cyrillic keyboard layouts.
_QUIT_KEYS = {27, ord('q'), ord('Q'), ord('\u0439'), ord('\u0419')}

# dt clamp. The upper bound is 0.3 s (not 0.1): YOLO on CPU gives 4-8 FPS, and
# with a tighter clamp ALL timers ran slower than real time -- the 3 s
# target-loss grace became 6 s, and the drone flew blind twice as long.
_DT_MIN, _DT_MAX = 1e-3, 0.3


class App:
    def __init__(self, source: FrameSource, detector=None, selector=None,
                 tracker=None, error_calculator=None, controller=None,
                 renderer: Renderer = None,
                 supervisor: FlightSupervisor = None, recorder=None):
        self._source = source
        self._renderer = renderer
        self._supervisor = supervisor
        self._recorder = recorder
        self._keys = TerminalKeys()   # works around the macOS window-focus problem

        self._pipeline = TrackingPipeline(
            detector, selector, tracker, error_calculator, controller,
            on_event=lambda msg: print(f"[app] {msg}"))

        self._renderer.set_mouse_callback(self._on_mouse)

        # We recognise a replay by the extra methods present on the source.
        self._replaying = hasattr(source, "replayed_detections")

        self._prev_time = None
        self._pending_click = None    # deferred click (x, y)
        self._pending_click_ttl = 0
        self._last_key = None         # key debounce
        self._last_key_time = 0.0

        self._controller = controller
        self._geometry_checked = False   # the setpoint check runs once

        self._stats = RunStats()
        # The metric's setpoint comes FROM THE CONTROLLER -- a single source of
        # truth, so the metric measures exactly what the system controls.
        self._follow = FollowScore(
            target_m=getattr(controller, "distance_target_m", 1.6),
            tolerance_m=3 * getattr(controller, "distance_deadzone_m", 0.25))

    # -- Input ---------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # We do NOT select immediately: the box may have vanished or shifted
            # at exactly that moment (lag, flicker). We try for several frames.
            self._pending_click = (x, y)
            self._pending_click_ttl = 10

    # -- The main loop -------------------------------------------------
    def run(self):
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break

                dt = self._frame_dt()
                self._check_geometry(frame)
                detections = (self._source.replayed_detections()
                              if self._replaying else None)

                result = self._pipeline.process(frame, dt, detections)

                self._seed_replay_target(result)
                self._resolve_pending_click(result)

                # The safety filter decides whether what the controller wants is
                # ALLOWED, and damps the velocities before the motors.
                sent = None
                if self._supervisor is not None:
                    sent = self._supervisor.update(result.command,
                                                   result.has_target, dt,
                                                   blind=result.blind)

                h, w = frame.shape[:2]
                self._report(result, sent)
                self._collect(result, sent, w, h)
                self._record(frame, dt, result, sent)
                self._draw(frame, result)

                if self._handle_input():
                    break
                if not self._renderer.is_open():
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C")
        finally:
            self._shutdown()

    # -- Loop steps (one thought each) ---------------------------------
    def _frame_dt(self):
        """On replay we take dt FROM THE RECORDING: on a fast run the real time
        between frames is tiny, and the Kalman filter and PID would operate on
        the wrong time scale."""
        if self._replaying:
            dt = self._source.replayed_dt()
        else:
            now = time.time()
            dt = 0.0 if self._prev_time is None else now - self._prev_time
            self._prev_time = now
        return max(_DT_MIN, min(dt, _DT_MAX))

    def _check_geometry(self, frame):
        """Once only: whether the distance setpoint is reachable at this frame size.

        WHY HERE AND NOT IN THE CONFIG. The check needs the FRAME HEIGHT, which
        is known only once the first frame has arrived -- the config is built long
        before that. We print loudly and fly on: in the air a warning is more
        useful than an exception on the first frame.

        This is exactly the check that was missing when the setpoint sat at
        0.75 m against a measurement limit of 1.33 m -- and for a whole flight
        the drone chased a person standing right in front of it."""
        if self._geometry_checked:
            return
        self._geometry_checked = True
        cfg = getattr(self._controller, "config", None)
        if cfg is None or not hasattr(cfg, "check_measurable"):
            return
        warning = cfg.check_measurable(frame.shape[0])
        if warning:
            print(f"[app] !!! {warning}")

    def _seed_replay_target(self, result):
        """On replay there is no mouse click -- we reproduce the operator's choice
        from the recorded target_id.

        WHY NOT ONCE. This began as a "seeded" flag for the whole run, and that
        broke the entire harness: as soon as the selector dropped the target (and
        it does, past its grace period) there was nobody to restore it, and the
        replay ran in LOST to the end. Measured: 12% target retention against 77%
        in that same live flight -- i.e. the harness was showing a different
        flight from the one recorded, and tuning on it was pointless.

        The right condition is not "have we seeded yet" but "is there a target
        NOW": the recording shows the operator was tracking at this moment, so we
        reproduce that every time the pipeline is left without a target."""
        if not self._replaying or self._pipeline.target_id is not None:
            return
        tid = self._source.replayed_target_id()
        if tid is None:
            return
        seed = next((d for d in result.detections if d.id == tid), None)
        if seed is not None:
            self._pipeline.select_target(seed)

    def _resolve_pending_click(self, result):
        if self._pending_click is None:
            return
        picked = self._pipeline.select_at(*self._pending_click, result.detections)
        self._pending_click_ttl -= 1
        if picked is not None:
            print(f"[app] target selected: #{picked.id} ({picked.name}) -- press G")
        elif self._pending_click_ttl <= 0:
            print("[app] click missed -- no box nearby")
        if picked is not None or self._pending_click_ttl <= 0:
            self._pending_click = None

    def _report(self, result, sent):
        if result.command is None:
            return
        e, c = result.error, result.command
        dist = ("  ---" if e.distance_m is None
                else f"{e.distance_m:4.2f}m" + ("!" if e.box_clipped else " "))
        line = (f"[{result.status:4}] ex={e.error_x:+7.0f} ey={e.error_y:+7.0f}"
                f" d={dist}   |   yaw={c.yaw:+6.1f}"
                f" vert={c.vertical:+6.1f} fwd={c.forward:+6.1f}")
        if sent is not None:
            line += (f"   -> MOTORS yaw={sent.yaw:+5.1f}"
                     f" vert={sent.vertical:+5.1f} fwd={sent.forward:+5.1f}")
        print(line)

    def _collect(self, result, sent, w, h):
        self._stats.add(result, sent, w, h)

        # A STALE FRAME DOES NOT ENTER THE SCORE AT ALL -- neither numerator nor
        # denominator.
        #
        # It used to land there as "there was no target", and the metric mixed two
        # different failures: "the system cannot track" and "the video gave us no
        # new frame". Measured on flight_20260926_224656: the stale frames alone
        # produced 66.8% "no target at all" and a FOLLOW SCORE of 4.8%, whereas on
        # the frames where the detector actually ran the target was found in 77%.
        # So the metric reported the rate of stalling rather than the quality of
        # tracking, and tuning the controller against it was pointless.
        #
        # (These three figures were measured live and cannot be reproduced from
        # the recordings, because the .mp4 re-encode destroys exact frame
        # duplication -- see perception/frame_quality.py. The reasoning is
        # unaffected: whatever the exact share, a stale frame says nothing about
        # tracking quality and must not enter the score.)
        #
        # The share of stale frames does not disappear -- RunStats reports it on a
        # line of its own, which is where it belongs.
        if result.frame_stale:
            return

        # The follow score counts ONLY a genuinely visible target: in PRED the
        # position is a Kalman hypothesis, and crediting it as a success would be
        # self-deception.
        #
        # We hand over the distance READY-MADE -- the same one the controller acts on.
        visible = result.status == VIS
        self._follow.add(result.center if visible else None,
                         result.error.distance_m if result.error else None,
                         w, h)

    def _record(self, frame, dt, result, sent):
        if self._recorder is None:
            return
        sup = self._supervisor
        # The CLEAN frame (before drawing) + the pipeline input + a command baseline.
        self._recorder.write(frame, dt, result.detections, self._pipeline.target_id,
                             sent=sent,
                             state=None if sup is None else sup.state,
                             battery=None if sup is None else sup.battery)

    def _draw(self, frame, result):
        fps = (1 / result.detect_seconds) if result.detect_seconds > 1e-6 else 999.0
        # We append the flight state to the tracking status -- so you can see at a
        # glance whether the drone is tracking or merely hovering, without
        # looking at the console.
        shown = result.status
        if self._supervisor is not None:
            shown = f"{result.status}|{self._supervisor.state}"
        # A broken frame is shown EXPLICITLY. Otherwise the screen only shows the
        # consequence ("the target vanished") and the cause -- garbage in the
        # video -- is easy to blame on YOLO.
        if result.frame_corrupt:
            shown += "|NOISE"
        if result.frame_stale:
            shown += "|STALE"
        self._renderer.draw(frame, result.detections, self._pipeline.target_id,
                            fps, shown, result.predicted)

    def _handle_input(self) -> bool:
        key = cv2.waitKey(1)
        typed = self._keys.get()      # keys from the terminal too (macOS)
        if typed:
            key = 27 if typed in ('\x1b', '\x03') else ord(typed.lower())
        return key != -1 and self._handle_key(key)

    def _handle_key(self, key) -> bool:
        """cv2.waitKey on macOS/Qt returns a code with extra HIGH BITS (1048679
        instead of 103 for 'g'), so we check both forms. Without this the keys
        stopped working as soon as focus moved to the OpenCV window."""
        codes = {key, key & 0xFF}

        # Debounce: one press can arrive twice (window + terminal). For G that
        # meant "armed and immediately disarmed" -- it looked broken.
        now = time.time()
        norm = key & 0xFF
        if norm == self._last_key and (now - self._last_key_time) < 0.4:
            return False
        self._last_key, self._last_key_time = norm, now

        if codes & _QUIT_KEYS:
            return True
        if self._supervisor is None:
            return False              # view-only mode -- flight disabled

        if ord('t') in codes:
            self._supervisor.takeoff()
        elif ord('l') in codes:
            self._supervisor.land()
        elif ord('g') in codes:
            self._supervisor.toggle()
        elif ord('x') in codes:
            self._supervisor.emergency()
        else:
            print(f"[app] unknown key (code {key}) -- T takeoff, G tracking, L land")
        return False

    def _shutdown(self):
        """Order matters: land the drone first, release resources second."""
        if self._supervisor is not None and self._supervisor.state != GROUNDED:
            print("Shutting down -- landing the drone ...")
            self._supervisor.land()
        if self._recorder is not None:
            self._recorder.close()
        self._keys.restore()
        self._source.release()
        cv2.destroyAllWindows()

        lines = self._stats.report()
        if lines:
            print("\n" + "-" * 60)
            for line in lines:
                print(line)
            for line in self._follow.report():
                print("  " + line)
