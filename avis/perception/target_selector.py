# Target choice and target "memory": holds which box we are chasing and does
# not lose it on partial occlusion (stepped behind a cupboard) or a brief
# complete disappearance.
#
# Two mechanisms:
#  1) VISIBILITY. An occluded target gives a smaller box, so visibility % =
#     area / reference. Below min_visibility the box centre is misleading, so we
#     hand control over to the prediction.
#  2) MEMORY + RE-ACQUISITION. A timer in seconds forgives a short absence; a
#     new box of the same class and size near the prediction is accepted as the
#     same target.
#
# WHY RE-ACQUISITION EARNS ITS KEEP. ByteTrack reissues track ids far more often
# than you would expect. Measured on flight_20261002_185914: 25 distinct target
# ids and 39 id changes across a single 218-second flight. Without the
# geometric re-acquisition below, every one of those would be a full target
# loss even though the person never left the frame.


def _point_box_distance(x, y, xyxy):
    """Distance from point (x, y) to a rectangle; 0 if the point is inside."""
    x1, y1, x2, y2 = xyxy
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


class TargetSelector:
    """Holds the target and does not lose it on occlusion or a brief absence."""

    def __init__(
        self,
        lost_grace_seconds=3.0,       # SECONDS of complete absence we forgive
        min_visibility=0.18,          # below this the target counts as occluded
        occlusion_debounce=3,         # FRAMES in a row the drop must persist
                                      #   before we believe in the occlusion
        reference_smoothing=0.1,      # EMA of the reference AREA
        reacquire_radius=120.0,       # base re-acquisition radius (px)
        reacquire_growth=80.0,        # +px of radius per second of absence
        reacquire_radius_max=220.0,   # radius CAP (was 400 -- half the frame,
                                      #   which made the drone grab bystanders)
    ):
        self._lost_grace_seconds = lost_grace_seconds
        self._min_visibility = min_visibility
        self._occlusion_debounce = occlusion_debounce
        self._reference_smoothing = reference_smoothing
        self._reacquire_radius = reacquire_radius
        self._reacquire_growth = reacquire_growth
        self._reacquire_radius_max = reacquire_radius_max

        self._target_id = None        # current target id; None = nothing chosen
        self._target_cls = None       # target class (so we do not latch onto another object)
        self._reference_area = None   # "full" target area (visibility reference)
        self._time_since_seen = 0.0   # seconds since the last reliable sighting
        self._occluded_frames = 0     # consecutive frames with a collapsed area

    @property
    def target_id(self):
        return self._target_id

    def select_at(self, x, y, detections, tolerance=70.0):
        """Choose a target by click (x, y). Forgiving of imprecision: if the
        click missed every box, take the nearest one within tolerance pixels."""
        inside = [d for d in detections
                  if d.xyxy[0] <= x <= d.xyxy[2] and d.xyxy[1] <= y <= d.xyxy[3]]
        if inside:
            # If the click landed inside several boxes at once (a person standing
            # next to a chair, boxes overlapping) take the one whose CENTRE is
            # nearest the click. This used to take the SMALLEST box by area, so
            # clicking a person beside a chair could select the chair.
            return self.select_target(min(
                inside,
                key=lambda d: (d.center[0] - x) ** 2 + (d.center[1] - y) ** 2,
            ))
        best, best_dist = None, tolerance
        for det in detections:
            dist = _point_box_distance(x, y, det.xyxy)
            if dist <= best_dist:
                best, best_dist = det, dist
        return self.select_target(best) if best is not None else None

    def select_target(self, det):
        """Assign a specific box as the target directly (a click, or replay)."""
        self._target_id = det.id
        self._target_cls = det.cls
        self._reference_area = det.area
        self._time_since_seen = 0.0
        self._occluded_frames = 0
        return det

    def clear(self):
        self._target_id = None
        self._target_cls = None
        self._reference_area = None
        self._time_since_seen = 0.0
        self._occluded_frames = 0

    # Visibility % = current area / reference. A sudden drop (occlusion) gives a
    # small percentage, while the slow change from walking away is absorbed into
    # the reference by the EMA.
    def _visibility(self, det):
        if not self._reference_area:
            return 1.0
        return det.area / self._reference_area

    def _update_reference(self, area):
        if self._reference_area is None:
            self._reference_area = area
        else:
            a = self._reference_smoothing
            self._reference_area = (1 - a) * self._reference_area + a * area

    # Find the same target under a NEW id -- the nearest box of the same class
    # and comparable size around the position the Kalman filter predicted.
    def _try_reacquire(self, detections, predicted_center):
        px, py = predicted_center
        radius = min(
            self._reacquire_radius + self._reacquire_growth * self._time_since_seen,
            self._reacquire_radius_max,
        )
        best, best_dist = None, radius
        for det in detections:
            if self._target_cls is not None and det.cls != self._target_cls:
                continue
            if self._reference_area and not (
                0.3 * self._reference_area <= det.area <= 3.0 * self._reference_area
            ):
                continue
            cx, cy = det.center
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist <= best_dist:
                best, best_dist = det, dist
        return best

    def resolve(self, detections, predicted_center=None, dt=0.0):
        """Returns (detection, lost):
          (det,  False) -- target reliably seen, or re-acquired;
          (None, True)  -- occlusion or brief loss: steer by the Kalman prediction;
          (None, False) -- target gone for good, or never chosen."""
        if self._target_id is None:
            return None, False

        current = next((d for d in detections if d.id == self._target_id), None)

        # Found by id -> check VISIBILITY.
        if current is not None:
            # WE DO NOT OVERWRITE THE TARGET CLASS. This line used to read
            #   self._target_cls = current.cls
            # and it was a bug: ByteTrack keeps the same track id while YOLO can
            # classify a person as a chair for a single frame -- and the target
            # class became "chair" FOREVER. Re-acquisition would then hunt for
            # chairs. The class is fixed once in select_target() and never changes.
            if self._visibility(current) >= self._min_visibility:
                self._time_since_seen = 0.0
                self._occluded_frames = 0
                self._update_reference(current.area)
                return current, False                 # reliably visible

            # OCCLUSION DEBOUNCE. A single frame with a collapsed area is usually
            # detector flicker, not a real occlusion. We only declare a loss when
            # the drop PERSISTS for several consecutive frames. Without this we
            # got spurious PRED states (especially while the distance changed
            # quickly), and in PRED control follows a prediction built on a stale
            # area.
            self._occluded_frames += 1
            if self._occluded_frames < self._occlusion_debounce:
                self._time_since_seen = 0.0
                return current, False                 # still trusting the box
            return self._tick_lost(dt)                # a persistent occlusion

        # Not found by id -> geometric re-acquisition around the prediction.
        if predicted_center is not None:
            candidate = self._try_reacquire(detections, predicted_center)
            if candidate is not None:
                # We refresh the id (the tracker issued a new one) but NOT the
                # class: that was fixed when the target was chosen, and the
                # candidate has already passed the class check.
                self._target_id = candidate.id
                self._time_since_seen = 0.0
                self._occluded_frames = 0
                self._update_reference(candidate.area)
                return candidate, False

        return self._tick_lost(dt)

    def _tick_lost(self, dt):
        self._time_since_seen += dt
        if self._time_since_seen > self._lost_grace_seconds:
            self.clear()
            return None, False
        return None, True
