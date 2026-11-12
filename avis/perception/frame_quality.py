# DETECTING BROKEN FRAMES (UDP packet loss in the Tello video stream).
#
# WHY THIS IS NEEDED. The Tello pushes H.264 over UDP with no retransmission:
# lose a packet and the decoder reconstructs the frame from whatever survived,
# smearing displaced macroblocks across the picture. Measured on real flights
# by replaying flight_20260926_224656 against its log (8993 frames):
#
#     detection on CLEAN frames     49%
#     detection on BROKEN frames     4%     <- the system is effectively blind
#
# A NOTE ON THESE TWO NUMBERS, because they are easy to over-read. The
# denominator is every frame, including the ones where the operator had simply
# walked out of view -- the log cannot tell "the detector failed" from "there
# was nobody to detect". So the absolute level is pessimistic for both; what
# the measurement establishes is the RATIO: a clean frame is ~12x more likely
# to yield a detection than a broken one. An earlier revision of this comment
# claimed 85% vs 6%, measured live on a subset where the person was known to be
# in frame; that subset no longer exists, so the figures above are the ones
# reproducible from the recordings.
#
# And the point that matters either way: during long target losses a
# disproportionate share of frames are broken, so a large part of what looks
# like "lost the target" is really GARBAGE ON SCREEN.
#
# WHY THE DETECTOR CANNOT FIX THIS. The tempting move is to lower the conf
# threshold so YOLO finds something even on a broken frame. That is the wrong
# road: on such a frame the object is not physically present, only artefacts
# are. A lower threshold buys false positives, not the target.
#
# THE RIGHT ANSWER is to separate two DIFFERENT states the system used to
# conflate:
#     "the target is gone"  -> this is INFORMATION, act on it (hover, land);
#     "I cannot see"        -> this is an ABSENCE of information, do NOT act.
# Absence of evidence is not evidence of absence. On a broken frame we simply
# stop the loss timer: time passes, but we have no grounds to call the target
# gone.
#
# HOW WE DETECT IT. H.264 cuts the frame into 16x16 macroblocks. When a packet
# is lost, neighbouring blocks come from different places, so seams appear on
# the 16px GRID that never occur in a natural image. We measure:
#
#     brightness step AT block boundaries / step INSIDE blocks
#
# The metric is dimensionless, so it does not depend on scene, lighting or
# resolution -- unlike "sharpness" or contrast.
#
# ABOUT THE ABSOLUTE RATIO VALUES. Do not calibrate the threshold against
# numbers measured from a .mp4 recording: FlightRecorder re-encodes through
# cv2.VideoWriter, which re-blocks the image and shifts the ratio (median 1.26
# on the re-encoded flight_20260926_224656 against roughly 1.1 on the live
# stream). That is exactly why the threshold below is RELATIVE to a baseline
# the class learns from the stream it is actually watching.
#
# A WARNING ABOUT FALSE METRICS. This was first hunted via blurriness
# (Laplacian variance) and not found: sharpness does not drop on broken frames,
# it RISES (the seams between blocks are new sharp edges). The grid is what has
# to be measured.

import numpy as np


class FrameQuality:
    """Whether a frame is broken by packet loss."""

    # threshold -- how many times the grid seams must exceed the normal level of
    #   this same stream. 1.35 was chosen with margin: on the live stream the
    #   median sits steadily around 1.11 while broken frames give 1.3-1.9+.
    # adapt -- how fast the baseline follows the stream (EMA). The baseline has
    #   to be adaptive: it differs slightly between 480p and 720p, and the Tello
    #   will serve either depending on whether it accepted our settings.
    # row_step -- we scan every 4th row, not all of them: the metric is averaged
    #   over hundreds of rows, so accuracy does not suffer while the cost drops
    #   fourfold. This runs ON EVERY FRAME, so cheapness is essential.
    def __init__(self, threshold=1.35, adapt=0.02, row_step=4, warmup=30):
        self._threshold = threshold
        self._adapt = adapt
        self._row_step = row_step
        self._warmup = warmup            # frames spent learning before judging
        self._baseline = None
        self._seen = 0
        self._prev_probe = None          # fingerprint of the previous frame (is_stale)

    # -- STALE FRAME ---------------------------------------------------
    #
    # A SECOND way to lose your sight, completely unlike corruption. A broken
    # frame is visible: it falls apart. A stale frame looks PERFECT -- it is
    # simply the PREVIOUS frame, processed a second time.
    #
    # WHERE IT COMES FROM. djitellopy keeps the last decoded frame in .frame and
    # hands it back on every request. When the decoder stalls (packet loss, a
    # network change) that field stops updating, while our loop keeps spinning
    # at full speed and reprocesses the same picture over and over.
    #
    # WHY IT IS MORE DANGEROUS THAN CORRUPTION. On a broken frame the detector
    # finds nothing, so the system at least knows it is blind. On a stale frame
    # it confidently sees the target (status VIS) and sends commands based on a
    # picture that is seconds old.
    #
    # MEASUREMENT CAVEAT -- READ BEFORE TRYING TO REPRODUCE. The staleness
    # figures quoted around this project (24% of frames on one flight, 47% of
    # AUTO frames on flight_20260926_224656, longest run 97 frames / 5.1 s) were
    # measured LIVE, in flight, and CANNOT be reproduced from the recordings:
    # FlightRecorder writes the .mp4 through cv2.VideoWriter, and lossy
    # re-encoding means two identical input frames decode to slightly different
    # pixels. Checked on flight_20260926_224656: 0% of consecutive .mp4 frames
    # are byte-identical while 87% are near-identical (mean |delta| < 0.5).
    # is_stale() keys on exact equality, so on a replay it reports almost
    # nothing. This is a real gap in the replay harness, not a bug here: stale
    # frames are the one failure mode the recordings do not preserve.
    #
    # HOW WE DETECT IT. Sample a few dozen pixels at a fixed stride and compare
    # against the previous frame. Two DIFFERENT frames from a camera never agree
    # on all probes: even a still scene carries sensor noise. The cost is
    # microseconds, because we compare dozens of numbers rather than a million.
    def is_stale(self, frame) -> bool:
        """True -- same frame as last time: there is no new information."""
        if frame is None or frame.ndim != 3:
            return False
        # Probe: every 37th pixel on both axes, one channel. 37 is prime so the
        # probe grid cannot land on a regular structure in the image.
        probe = frame[::37, ::37, 1].copy()
        stale = (self._prev_probe is not None
                 and self._prev_probe.shape == probe.shape
                 and np.array_equal(self._prev_probe, probe))
        self._prev_probe = probe
        return stale

    @staticmethod
    def _grid_ratio(frame, row_step):
        """Ratio of "seams on the 16px grid" to "steps inside blocks"."""
        # Take ONE channel rather than cvtColor to grey: the artefacts show in
        # any channel, and we avoid converting the whole frame for nothing.
        g = frame[::row_step, :, 1].astype(np.float32)
        w = g.shape[1]
        n = (w - 16) // 16
        if n < 4:
            return None
        edge_cols = 16 + 16 * np.arange(n)      # columns lying ON a block boundary
        mid_cols = 8 + 16 * np.arange(n)        # columns INSIDE a block
        edge = np.abs(g[:, edge_cols] - g[:, edge_cols - 1]).mean()
        inner = np.abs(g[:, mid_cols] - g[:, mid_cols - 1]).mean()
        if inner < 1e-6:                        # flat patch (e.g. a black frame)
            return None
        return float(edge / inner)

    def is_corrupt(self, frame) -> bool:
        """True -- this frame cannot be trusted: it shows garbage, not a scene."""
        if frame is None or frame.ndim != 3:
            return False
        ratio = self._grid_ratio(frame, self._row_step)
        if ratio is None:
            return False

        if self._baseline is None:
            self._baseline = ratio
        self._seen += 1

        # Until the baseline is established we accuse nobody, we only learn.
        if self._seen <= self._warmup:
            self._baseline = (1 - self._adapt) * self._baseline + self._adapt * ratio
            return False

        corrupt = ratio > self._baseline * self._threshold

        # The baseline updates ONLY on clean frames. Otherwise a long run of
        # broken frames would drag it up to their own level, the detector would
        # "get used to" the garbage and stop seeing it -- the classic adaptive
        # threshold trap.
        if not corrupt:
            self._baseline = (1 - self._adapt) * self._baseline + self._adapt * ratio
        return corrupt
