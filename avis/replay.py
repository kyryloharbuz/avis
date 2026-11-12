# THE REPLAY HARNESS: record a flight once, then play it back instead of a live
# camera as many times as you like, tuning coefficients against an IDENTICAL
# input.
#
# Why: every real flight is different (target, light, wind), so you cannot tell
# whether an improvement came from a new kp or simply from an easier flight. A
# replay gives you a CONTROLLED experiment: same input, change one thing, see
# the clean effect.
#
# IMPORTANT: we store the pipeline's INPUT (frames + detections + dt), not its
# OUTPUT (commands). The commands are recomputed on playback with the new
# coefficients -- that is the whole point.
#
# KNOWN GAP IN THE HARNESS, worth stating because it silently biases results.
# Two things the recording does NOT preserve:
#   * STALE FRAMES. The .mp4 is written through cv2.VideoWriter, so lossy
#     re-encoding means two identical input frames decode to slightly different
#     pixels. Measured on flight_20260926_224656: 0% of consecutive frames are
#     byte-identical, 87% are near-identical. FrameQuality.is_stale() keys on
#     exact equality, so on replay it reports almost nothing however bad the live
#     stream was.
#   * THE CLOSED LOOP. The frames are fixed, so the drone's response cannot
#     change the next frame. A replay therefore shows how the command profile
#     changes, not how the flight changes. The follow score of a replay is
#     essentially a property of the recording, not of the current tuning -- do
#     not read a gain change out of it.
#
# Recording format (in recordings/, which is in .gitignore):
#   NAME.mp4    -- the frames themselves;
#   NAME.jsonl  -- one line per frame: {frame, dt, target_id, detections[]}.
# Recorder and ReplaySource live in one file on purpose -- so the data schema is
# in a single place and cannot drift out of sync.
#
# ReplaySource is the THIRD FrameSource implementation (alongside Webcam and
# Tello). App does not change: it simply reads frames. The two extra methods
# (replayed_dt, replayed_detections) are used by App IF they exist (duck
# typing) -- that is what enables "log replay": detections come from the file,
# YOLO never runs, and PID/Kalman tuning goes hundreds of times faster than real
# time.

import json
import os

import cv2

from avis.models import Detection
from avis.perception.frame_source import FrameSource


class FlightRecorder:
    """Writes frames (.mp4) and the pipeline input (.jsonl) during a live run."""

    def __init__(self, name, out_dir="recordings", fps=30):
        os.makedirs(out_dir, exist_ok=True)
        self._video_path = os.path.join(out_dir, f"{name}.mp4")
        self._log_path = os.path.join(out_dir, f"{name}.jsonl")
        self._fps = fps
        self._writer = None                 # lazy init: we learn the size from frame 1
        self._log = open(self._log_path, "w")
        self._frame_idx = 0
        print(f"[rec] recording to {self._video_path} + {self._log_path}")

    def write(self, frame, dt, detections, target_id,
              sent=None, state=None, battery=None):
        """Record one frame. Call this with a CLEAN frame (before boxes are
        drawn), otherwise the boxes will be burned into the replay video.

        sent/state/battery are the BASELINE of that flight: which commands
        actually reached the motors, what state the supervisor was in, what the
        battery was. On replay the commands are recomputed from scratch; these
        exist so there is something to compare against ("it got sharper -- by how
        much exactly?")."""
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._video_path, fourcc, self._fps, (w, h))
        self._writer.write(frame)
        h, w = frame.shape[:2]
        record = {
            "frame": self._frame_idx,
            "dt": round(dt, 5),
            "target_id": target_id,
            # THE FRAME SIZE. Needed by offline analysis (score.py), which reads
            # only the .jsonl and never opens the video. Without it we had to
            # assume 960x720 while the Tello serves 648x478 -- and the "target in
            # the central third" check was computed against a frame that does not
            # exist.
            "frame_w": w,
            "frame_h": h,
            "detections": [
                {
                    "id": d.id,
                    "xyxy": [round(v, 1) for v in d.xyxy],
                    "conf": round(d.conf, 3),
                    "cls": d.cls,
                    "name": d.name,
                }
                for d in detections
            ],
            "state": state,        # GROUNDED / HOVER / AUTO on this frame
            "battery": battery,    # % (cached, no extra queries to the drone)
            # The commands that REACHED THE MOTORS on that flight (None = flight disabled).
            "sent": None if sent is None else {
                "yaw": round(sent.yaw, 1),
                "vertical": round(sent.vertical, 1),
                "forward": round(sent.forward, 1),
            },
        }
        self._log.write(json.dumps(record) + "\n")
        self._frame_idx += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
        self._log.close()
        print(f"[rec] saved {self._frame_idx} frames")


class ReplaySource(FrameSource):
    """A frame source backed by a recording. Also serves the recorded
    dt/detections/target."""

    def __init__(self, name, out_dir="recordings"):
        video = os.path.join(out_dir, f"{name}.mp4")
        log = os.path.join(out_dir, f"{name}.jsonl")
        if not (os.path.exists(video) and os.path.exists(log)):
            raise FileNotFoundError(
                f"No recording '{name}' in {out_dir}/ (need {name}.mp4 and {name}.jsonl)"
            )
        self._cap = cv2.VideoCapture(video)
        with open(log) as f:
            self._log = [json.loads(line) for line in f if line.strip()]
        self._idx = -1
        self._current = None
        print(f"[replay] '{name}': {len(self._log)} frames")

    def read(self):
        ok, frame = self._cap.read()
        nxt = self._idx + 1
        if not ok or nxt >= len(self._log):
            return None                      # end of the recording
        self._idx = nxt
        self._current = self._log[self._idx]  # the log line for this frame
        return frame

    # --- App uses these three ONLY in replay mode (via hasattr) ---
    def replayed_dt(self):
        """The dt THAT THE FLIGHT HAD. Critical: measuring real time between
        frames during a replay gives a tiny value (we run fast) and would break
        the Kalman filter and the PID. So we serve the stored dt, and the filter
        'believes' time is passing as it did in flight even though everything is
        running many times faster."""
        return self._current["dt"]

    def replayed_target_id(self):
        """Which target the operator was tracking in this frame (to reproduce the
        choice -- there is no mouse click on replay)."""
        return self._current.get("target_id")

    def baseline(self):
        """The commands and state of THAT flight (the baseline). None if the
        recording is in the old format or the flight was disabled. Needed to
        compare "before/after" on identical input."""
        return {
            "sent": self._current.get("sent"),
            "state": self._current.get("state"),
            "battery": self._current.get("battery"),
        }

    def replayed_detections(self):
        """This frame's recorded detections as Detection objects (YOLO is not run)."""
        return [
            Detection(
                id=d["id"], xyxy=tuple(d["xyxy"]), conf=d["conf"],
                cls=d["cls"], name=d.get("name", ""),
            )
            for d in self._current["detections"]
        ]

    def release(self):
        self._cap.release()
