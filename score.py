# Evaluate recorded flights with an Andon Labs Drone-Bench-style follow metric.
# The script reads JSONL logs only, which makes scoring fast and resilient to a
# partially written video file.
#
# Usage:
#   .venv/bin/python score.py
#   .venv/bin/python score.py flight_2026...
#   .venv/bin/python score.py --target 1.2

import glob
import json
import os
import sys

from avis.control.follow.config import FollowConfig
from avis.distance import DISTANCE_K, meters_from_height, nearest_measurable_m
from avis.metrics import HUMAN_BASELINE, FollowScore

# Fallback when neither the log nor the video exposes frame dimensions.
DEFAULT_W, DEFAULT_H = 960, 720


def frame_size(log_path, first_record):
    """Return recording dimensions from the log, video header, or fallback."""
    w, h = first_record.get("frame_w"), first_record.get("frame_h")
    if w and h:
        return w, h                    # current logs store dimensions directly

    # Older logs obtain dimensions from the video header without frame decoding.
    video = log_path[:-6] + ".mp4"
    if os.path.exists(video):
        try:
            import cv2
            cap = cv2.VideoCapture(video)
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vw > 0 and vh > 0:
                return vw, vh
        except Exception:
            pass
    return DEFAULT_W, DEFAULT_H


def score_file(path, target_m, tolerance_m, auto_only=False):
    """Calculate the follow score for one recording.

    With ``auto_only=True``, score only frames in the autonomous-flight state.
    """
    fs = FollowScore(target_m=target_m, tolerance_m=tolerance_m)
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return fs
    w, h = frame_size(path, records[0])
    for r in records:
        if auto_only and r.get("state") != "AUTO":
            continue
        tid = r.get("target_id")
        # Locate the selected target among the frame detections.
        target = None
        if tid is not None:
            target = next((d for d in r["detections"] if d["id"] == tid), None)
        if target is None:
            fs.add(None, None, w, h)        # no selected target
            continue
        x1, y1, x2, y2 = target["xyxy"]
        # Use the same height-based distance estimate as the flight controller.
        distance_m = meters_from_height(y2 - y1)
        fs.add(((x1 + x2) / 2, (y1 + y2) / 2), distance_m, w, h)
    return fs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # Keep the metric setpoint aligned with the controller configuration.
    cfg = FollowConfig()
    target_m = cfg.distance_target_m
    if "--target" in sys.argv:
        target_m = float(sys.argv[sys.argv.index("--target") + 1])
    # Use the same tolerance formula as App: three deadzones.
    tolerance_m = 3 * cfg.distance_deadzone_m

    files = ([os.path.join("recordings", a + ".jsonl") for a in args] if args
             else sorted(glob.glob("recordings/*.jsonl")))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("No recordings found in recordings/")
        return

    print("\nFOLLOW SCORE — target centred and within the requested distance")
    print(f"Andon Labs Drone-Bench human baseline: {100 * HUMAN_BASELINE['Follow']:.1f}%")
    print(f"Target distance: {target_m:.2f} ± {tolerance_m:.2f} m")
    # The measurement limit must use the real frame height of the recording.
    with open(files[0]) as fh:
        first = next((json.loads(x) for x in fh if x.strip()), {})
    _, probe_h = frame_size(files[0], first)
    print(f"Distance from box height, K={DISTANCE_K:.0f} (no calibration)."
          f" Measurement limit: {nearest_measurable_m(probe_h):.2f} m for"
          f" {probe_h} px\n")
    print(f"{'recording':<22} {'frames':>7} {'FULL RUN':>12} {'AUTONOMOUS':>18}")
    print("-" * 64)

    tot_g = tot_n = auto_g = auto_n = 0
    for path in files:
        name = os.path.basename(path)[:-6]
        full = score_file(path, target_m, tolerance_m)
        auto = score_file(path, target_m, tolerance_m, auto_only=True)
        a_txt = f"{100 * auto.score:.1f}%" if auto.frames else "—"
        print(f"{name.replace('flight_', ''):<22} {full.frames:>7} "
              f"{100 * full.score:>11.1f}% {a_txt:>18}")
        tot_g += full.good
        tot_n += full.frames
        auto_g += auto.good
        auto_n += auto.frames

    print("-" * 64)
    print(f"{'TOTAL':<22} {tot_n:>7} {100 * tot_g / max(tot_n, 1):>11.1f}% "
          f"{100 * auto_g / max(auto_n, 1):>17.1f}%")
    print("\n  FULL RUN   — includes frames before target selection and on the ground")
    print("  AUTONOMOUS — includes only AUTO-state frames")
    print("\n  This implementation does not detect collisions and estimates distance")
    print("  from bounding-box height rather than a direct range sensor.")


if __name__ == "__main__":
    main()
