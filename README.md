# AVIS

> Visual follow-me control for a Ryze Tello: detect a chosen object, keep it in frame, and turn that estimate into safety-limited motion commands.

AVIS is a Python prototype for autonomous visual following. It combines YOLO detections, ByteTrack identities, a frame-quality gate for the lossy Wi-Fi stream, Kalman prediction, independent PD control axes, and a flight supervisor that has the final say before any command reaches the drone.

![AVIS steering by a Kalman prediction while the target is occluded](docs/assets/tracking-predict.jpg)

## What it does

- Detects objects with Ultralytics YOLO — one chosen COCO class or all of them — and keeps an identity with ByteTrack.
- Lets the operator choose the target with a click. The Kalman filter was added for Wi-Fi latency — every decision is made on an image that is already old, so the system tracks the target's velocity instead of its last known box. The same estimate holds the target when it briefly disappears.
- Treats a corrupt or repeated frame as *absence of information*, not absence of the target: a stale frame never reaches the detector, the target-loss timer pauses, and the last command is held briefly instead of being zeroed.
- Steers yaw and distance with independent PD axes; forward motion is reduced until the drone faces the target. A vertical axis is built and tested but ships disabled — the Tello holds altitude on its own, and closing that loop over ~250 ms of Wi-Fi latency turned its slow level-learning into an oscillation.
- Separates the controller's *desired* command from the command the flight supervisor actually allows.
- Records the raw video, detections, timing, and baseline commands, then replays the same flight for repeatable tuning.

## Demo

These frames are from real indoor Tello runs. The red box is the selected target, the yellow reticle is the Kalman hypothesis during an occlusion, and the banner shows an actual command from the recording.

| Predict through occlusion | Correct yaw | Adjust distance |
| --- | --- | --- |
| ![Kalman prediction](docs/assets/tracking-predict.jpg) | ![Yaw correction](docs/assets/tracking-yaw.jpg) | ![Distance adjustment](docs/assets/tracking-range.jpg) |

## Architecture

```mermaid
%%{init: {"themeVariables": {"fontSize": "16px"}}}%%
flowchart TD
    subgraph P["perception"]
        direction LR
        camera["Camera / replay"] --> gate["Frame quality gate\ncorrupt / stale"]
        gate -->|"frame carries\ninformation"| detection["YOLO + ByteTrack"]
        detection --> selection["Target selection\nVIS / PRED / LOST"]
        selection --> estimation["Kalman filter +\ndistance estimate"]
    end
    subgraph C["control"]
        direction LR
        axes["Yaw / vertical / distance\ncontrol axes"] --> safety["Flight supervisor\nsafety limits"] --> tello["Tello"]
    end
    estimation --> axes
    gate -.->|"no new information —\nhold the last command, freeze the loss timer"| safety
    camera --> recorder["Flight recorder"] --> replay["Repeatable replay"]
    replay --> detection
```

**Why the gate is in the picture.** The Tello streams H.264 over UDP with no retransmission, and `djitellopy` re-serves the last decoded frame whenever its decoder stalls. Treating either case as "target lost" used to zero the motors in **52% of control cycles** on one measured flight (3474 of 6625 AUTO frames — 86.6 s of standing still). Corrupt frames are recognised by their 16×16 macroblock-grid artifacts and still run through the detector, with the loss timer paused; repeated frames are caught by an exact-equality pixel probe and skip detection entirely. Neither counts as evidence that the target is gone.

The detailed component map, data flow, and design decisions are in [the architecture guide](docs/architecture.md).

## Quick start

**Prerequisites:** Python 3.12, a webcam for local testing, or a Ryze Tello connected to the host computer's Wi-Fi. Use an isolated, obstacle-free area for all flight tests.

```bash
git clone https://github.com/kyryloharbuz/avis.git
cd avis
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start with the webcam-only mode:

```bash
python main.py --person
```

Then progress through the Tello modes deliberately:

```bash
python main.py --tello                 # video stream only; no flight commands
python main.py --tello --fly --person  # flight control, forward motion enabled
```

Click the desired detection in the video window. In flight mode, press `T` to take off and `G` to arm/disarm autonomous tracking — a `G` pressed while the takeoff is still running is remembered and applied once the drone reaches hover. Press `L` to land; `X` is the emergency-stop command. `Esc` or `Q` ends the application and triggers the normal shutdown path.

> **Safety note:** this is an experimental controller for a real aircraft. Keep a clear landing area, maintain visual line of sight, start with forward motion disabled (`--no-forward`), and be ready to land manually. The software is not a substitute for responsible operation or local flight rules.

## Replay a recorded flight

Each live run records a video and a JSONL log into `recordings/` (ignored by Git because recordings are large and may contain private footage). Replay uses the recorded detections and timing, so it avoids running YOLO and makes control tuning repeatable:

```bash
python main.py --replay flight_20261004_182119
python main.py --replay flight_20261004_182119 --headless
```

The name is whatever a previous live run printed on start; the repository ships no recordings. The second command is useful for fast, non-interactive runs. Replay is an analysis tool, not a physics simulator: it recomputes the command profile against fixed video and cannot show how a different command would have changed the real drone trajectory.

## Test

Two regression suites, both plain assertions with no test framework. The first covers value objects, tracking behaviour, safety states, replay and integration paths; the second covers the control axes and replays real recordings to check that the commands are strong enough to move the aircraft:

```bash
.venv/bin/python tests/test_all.py      # 138 checks
.venv/bin/python tests/test_follow.py   # 117 checks, replays recordings if present
```

## Project layout

```text
avis/
  perception/    # frame sources, frame-quality gate, YOLO/ByteTrack, target selection, Kalman filter
  control/       # PD control axes, flight state machine, drone adapters
  view/          # OpenCV rendering and keyboard input
  app.py         # orchestration: frame loop and I/O wiring
  pipeline.py    # pure computation: frame -> detections -> target -> command
  distance.py    # box height <-> metres and the measurement limit
  models.py      # Detection, ControlError, Command value objects
  metrics.py     # Follow score in the Andon Labs Drone-Bench style
  stats.py       # run statistics: saturation, jitter, frame health
  replay.py      # recording format and replay source
docs/
  architecture.md
  usage.md
tests/           # test_all.py, test_follow.py
main.py          # composition root and command-line flags
score.py         # scores recorded flights from their JSONL logs
tello_check.py   # pre-flight diagnostics; never commands takeoff
tello_fly.py     # manual keyboard control, no autonomy
```

## Documentation

- [Architecture and design rationale](docs/architecture.md)
- [Operating guide and command reference](docs/usage.md)

## Current scope

AVIS is an experimental project, tested with indoor recordings and a Tello. Its replay mode is intentionally a controlled input harness rather than a full flight simulator. Large recordings, model weights, calibration data, and local run artifacts stay out of the repository by design.
