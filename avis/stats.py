# RUN STATISTICS -- accumulation and reporting.
#
# Extracted from App for the same reason as the pipeline: App should orchestrate,
# not count. This used to be a ten-key dictionary, an accumulate method and a
# print method -- three different things inside a class whose job was to drive
# the loop.
#
# The classes involved:
#   RunStats -- states, error, SATURATION and JITTER (tuning metrics)
#   FollowScore lives separately in metrics.py -- it is the Andon Labs QUALITY
#   metric, with a different role: not "how is it tuned" but "how well do we
#   track".

from avis.pipeline import LOST, PRED, VIS


class RunStats:
    """Accumulates run figures for an HONEST comparison between tunings."""

    def __init__(self):
        self.frames = 0
        self.by_status = {VIS: 0, PRED: 0, LOST: 0}
        self._sum_ex = 0.0
        self._sum_ey = 0.0
        self._tracked = 0
        # Saturation: how often the supervisor clipped the command. We do not
        # need to know the limits -- the fact of clipping IS the saturation.
        # Note this counts BOTH causes together: the ceiling and the slew ramp.
        # If you are tuning, separate them before drawing conclusions -- on
        # flight_20261002_185914 the forward axis was clipped by the ceiling in
        # 12% of frames and by the ramp in 0%, while yaw was 35% and 20%
        # respectively. One number cannot tell you which knob to turn.
        self._sat = {"forward": 0, "yaw": 0, "vertical": 0}
        self._cmd_frames = 0
        # Jitter: how far the command jumped since the previous frame. High
        # jitter means the drone twitches (kp too high, or the deadzone too
        # narrow).
        self._jitter = 0.0
        self._jitter_n = 0
        self._prev_sent = None
        # Broken frames (UDP packet loss). This is an indicator of LINK QUALITY,
        # not of tuning: if it is high, no coefficient will help -- you need to
        # clear the 2.4 GHz interference or fly closer to the laptop.
        self._corrupt = 0
        # Stale frames -- the decoder had nothing new, so we processed the
        # previous one twice. An indicator of STREAM HEALTH, not of tuning: a
        # high value means the video is stalling and control is following an old
        # picture. NOTE: on a replay this will read near zero regardless of the
        # truth, because the .mp4 re-encode destroys exact frame duplication --
        # see perception/frame_quality.py.
        self._stale = 0

    def add(self, result, sent, frame_w, frame_h):
        self.frames += 1
        self.by_status[result.status] = self.by_status.get(result.status, 0) + 1
        if getattr(result, "frame_corrupt", False):
            self._corrupt += 1
        if getattr(result, "frame_stale", False):
            self._stale += 1

        if result.center is not None:
            self._tracked += 1
            self._sum_ex += abs(result.center[0] - frame_w / 2)
            self._sum_ey += abs(result.center[1] - frame_h / 2)

        cmd = result.command
        if cmd is None or sent is None:
            return
        self._cmd_frames += 1
        eps = 0.5
        for axis in self._sat:
            if abs(getattr(sent, axis)) + eps < abs(getattr(cmd, axis)):
                self._sat[axis] += 1
        if self._prev_sent is not None:
            self._jitter += sum(abs(getattr(sent, a) - getattr(self._prev_sent, a))
                                for a in ("yaw", "vertical", "forward"))
            self._jitter_n += 1
        self._prev_sent = sent

    def report(self):
        """Report lines. These are the numbers to compare between replay runs."""
        if self.frames == 0:
            return []
        n = self.frames
        tracked = max(self._tracked, 1)
        cf = self._cmd_frames
        lines = [
            f"SUMMARY: {n} frames   "
            f"VIS={self.by_status[VIS]} PRED={self.by_status[PRED]} LOST={self.by_status[LOST]}",
            f"  target retention: {100 * (self.by_status[VIS] + self.by_status[PRED]) / n:.0f}%",
            f"  mean error      : ex={self._sum_ex / tracked:.0f}px  "
            f"ey={self._sum_ey / tracked:.0f}px",
        ]
        if self._corrupt:
            pct = 100 * self._corrupt / n
            note = ("— link healthy" if pct < 3 else
                    "— interference visible" if pct < 10 else
                    "— LINK POOR; likely primary cause of target loss")
            lines.append(f"  corrupt frames : {pct:.0f}%  {note}")
        if self._stale:
            pct = 100 * self._stale / n
            note = ("— stream stable" if pct < 5 else
                    "— video stalls visible" if pct < 15 else
                    "— VIDEO FROZEN; control uses an old image")
            lines.append(f"  stale frames   : {pct:.0f}%  {note}")
        if cf:
            lines.append(
                f"  saturation     : forward {100 * self._sat['forward'] / cf:.0f}%  "
                f"yaw {100 * self._sat['yaw'] / cf:.0f}%  "
                f"vertical {100 * self._sat['vertical'] / cf:.0f}%")
            if self._jitter_n:
                lines.append(f"  command jitter : {self._jitter / self._jitter_n:.1f} units/frame")
        return lines
