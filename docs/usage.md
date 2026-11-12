# Operating AVIS

This page describes the supported run modes and the minimum safe sequence for a real Tello flight.

## Before a flight

1. Choose an uncluttered indoor space and make sure the intended landing area is clear.
2. Charge the Tello and connect the computer to the drone's Wi-Fi network.
3. Start with the video-only command. Verify that the picture is live and that detections look plausible.
4. Test the follow logic in replay before changing controller parameters or enabling forward motion.
5. Keep the drone in line of sight and keep a manual landing path clear at all times.

The project rejects takeoff below 30% battery, lands below 15% battery, hovers after a short target loss, and may land after a prolonged loss *only after a target has been acquired*. These guards reduce risk; they do not make the system safe for unattended operation.

## Run modes

| Command | Purpose | Sends motor commands? |
| --- | --- | --- |
| `python main.py` | Webcam with all detectable object classes | No |
| `python main.py --person` | Webcam, person class only | No |
| `python main.py --tello` | Tello video stream for pre-flight checks | No |
| `python main.py --tello --fly --person --no-forward` | Tello tracking with forward motion disabled | Yes, yaw only by default |
| `python main.py --tello --fly --person` | Full Tello follow mode | Yes |
| `python main.py --replay NAME` | Recompute control against `recordings/NAME.*` | No real drone |
| `python main.py --replay NAME --headless` | Fast replay without a window | No real drone |

Every live run records by default. Use `--record my_run` to choose a name or `--no-record` to disable recording.

## Flight sequence

1. Run the video-only mode and select the target by clicking its box.
2. Restart in flight mode when the scene and selection are working as expected.
3. Press `T` to request takeoff. The program keeps the visual loop responsive while the drone takes off.
4. After takeoff, the state is `HOVER`. The controller does not move the drone automatically.
5. Press `G` to arm autonomous tracking. There is a short settling period after takeoff.
6. Use `G` again to disarm tracking and remain airborne in hover mode. Press `L` to land.

`G` pressed during takeoff is remembered and arms tracking once the drone reaches hover. This is useful but should not replace visually confirming that the aircraft is stable first.

## Controls

| Key | Action |
| --- | --- |
| Mouse click | Select the detected target closest to the click |
| `T` | Take off |
| `G` | Arm/disarm autonomous tracking |
| `L` | Land |
| `X` | Emergency: stop motors immediately |
| `Esc` or `Q` | Quit and follow the normal shutdown/landing path |

Keyboard input is read both from the OpenCV window and from the terminal because window focus can be unreliable on macOS.

The keys are listed in capitals for readability, but they are matched in **lower case**. The terminal path lower-cases what it reads, so Shift makes no difference there; the OpenCV window passes the raw code through, so a capital `T` arrives as a different code and is reported as an unknown key. Keep Caps Lock off when driving the window.

## Recordings and replay

A live recording consists of two files with the same name:

```text
recordings/NAME.mp4    # clean camera frames
recordings/NAME.jsonl  # per-frame timing, detections, target, state, baseline commands
```

The JSONL file stores pipeline inputs, not just outputs. On replay, AVIS uses the original detections and frame timing while recalculating Kalman/PID/controller output with the current parameters. This makes comparisons meaningful when changing one setting at a time.

The limitation is important: recorded frames are fixed. Replay can compare command profiles, but it cannot predict a new physical trajectory from a new command.

## Tuning safely

Use replay to tune a single parameter and keep a note of the recording name and change. Only promote a tuning change to a live flight after the replay output looks reasonable.

The highest-risk setting is forward motion. It is constrained by both the `AlignmentGate` (which reduces forward motion when the target is off-centre) and `FlightSupervisor` limits. Do not raise its ceiling casually: a fast forward pitch can degrade the camera image and make detection less reliable.
