# Application entry point: `python main.py`.
# The composition root is the only place that selects concrete implementations.

from datetime import datetime

# Import through package facades.
from avis.app import App
from avis.control import ErrorCalculator, FollowController
from avis.perception import Detector, KalmanFilter, TargetSelector, WebcamSource
from avis.view import CvRenderer, HeadlessRenderer


def _build_tello(fly: bool, forward: bool):
    """Build the Tello frame source and optional flight supervisor.

    A single Tello instance serves both video and commands to avoid a UDP-port
    conflict. The import stays local so webcam mode does not load Tello extras.
    """
    from djitellopy import Tello

    from avis.control.drone import TelloDrone
    from avis.control.flight import FlightSupervisor
    from avis.perception.tello_source import TelloSource

    tello = Tello()
    source = TelloSource(tello=tello)          # shared instance
    supervisor = None
    if fly:
        supervisor = FlightSupervisor(TelloDrone(tello), enable_forward=forward)
    return source, supervisor


def _build_replay(name: str):
    """Build replay mode with recorded detections and a null drone.

    Replay starts armed so the full command path is visible without loading
    YOLO or connecting to an aircraft.
    """
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.replay import ReplaySource

    source = ReplaySource(name)
    supervisor = FlightSupervisor(NullDrone(), enable_forward=True, start_armed=True)
    return source, supervisor


# Sentinel used when recording is explicitly disabled with --no-record.
_NO_RECORD = object()


def build_app(use_tello=False, fly=False, forward=False,
              record_name=None, replay_name=None, person_only=False,
              headless=False) -> App:
    """Build the application with webcam, Tello, or replay input."""
    # Class 0 excludes furniture and other non-target COCO classes.
    detector = Detector(classes=[0] if person_only else None)
    recorder = None

    if replay_name:
        source, supervisor = _build_replay(replay_name)
        detector = None            # replay provides recorded detections
    elif use_tello:
        source, supervisor = _build_tello(fly, forward)
    else:
        source, supervisor = WebcamSource(), None

    # Record all live inputs by default for reproducible debugging.
    if not replay_name and record_name != _NO_RECORD:
        from avis.replay import FlightRecorder
        if not record_name:
            record_name = "flight_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder = FlightRecorder(record_name)

    return App(
        source=source,
        supervisor=supervisor,
        detector=detector,
        selector=TargetSelector(),
        tracker=KalmanFilter(),
        # Distance uses bounding-box height; no calibration dependency is needed.
        error_calculator=ErrorCalculator(),
        controller=FollowController(),
        # Headless mode accelerates replay by skipping rendering.
        renderer=HeadlessRenderer() if headless else CvRenderer(),
        recorder=recorder,
    )


def _arg_value(argv, flag):
    """Return the value following a command-line flag."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


if __name__ == "__main__":
    # Modes:
    #   python main.py                         webcam
    #   python main.py --tello                 Tello video only
    #   python main.py --tello --fly           autonomous following
    #   python main.py --replay flight_001     replay a recording
    # Live recording uses a timestamp by default; set --record NAME or --no-record.
    import sys
    if "--no-record" in sys.argv:
        rec = _NO_RECORD
    else:
        rec = _arg_value(sys.argv, "--record")
    # Pre-flight errors are expected operational failures; show a concise message.
    try:
        build_app(
            use_tello="--tello" in sys.argv,
            fly="--fly" in sys.argv,
            forward="--no-forward" not in sys.argv,
            record_name=rec,
            replay_name=_arg_value(sys.argv, "--replay"),
            person_only="--person" in sys.argv,   # detect class 0 only
            headless="--headless" in sys.argv,    # replay without a window
        ).run()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n[!] {e}")
        sys.exit(1)
