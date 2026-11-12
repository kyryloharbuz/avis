# Object detector built on YOLO plus the ByteTrack tracker.
#
# NO INTERFACE HERE (the middle ground): there is one implementation and no
# second one is expected. The class carries the clean generic name `Detector`.
# If a second implementation ever appears we extract an interface in two
# minutes (add an ABC and rename this to YoloDetector).

from ultralytics import YOLO

from avis.models import Detection


class Detector:
    # conf -- minimum confidence, 0.35. The ultralytics default of 0.25 is too
    #   low: weak detections reached the pipeline and caused the
    #   "person <-> chair" confusion. 0.5 was the first fix and it overshot --
    #   measured on a real recording (120 frames), 0.35 yields +2.5% detections
    #   over 0.5 with no loss of quality (median confidence 0.81, i.e. there are
    #   barely any weak false positives to cut in the first place).
    # classes -- which COCO classes to detect at all. None = all 80.
    #   [0] = only the chosen target class; chairs, sofas and plants then
    #   physically cannot become the target.
    # imgsz stays at 320: the same measurement showed 480/640 buy only +5%
    #   detections for DOUBLE the time -- the limit is not resolution, it is the
    #   video itself (packet loss, motion blur).
    #
    # THE RESOLUTION CLAIM WAS RE-MEASURED AND IT HOLDS -- in fact more strongly
    # than stated above. Taking 150 frames from flight_20261002_185914 on which
    # the live system found nothing at all, and re-running this model offline:
    #     imgsz=320 conf=0.35 (current) -> a person found on 19% of them
    #     imgsz=480 conf=0.35           -> 17%
    #     imgsz=640 conf=0.35           -> 17%
    #     imgsz=320 conf=0.20           -> 31%
    #     imgsz=640 conf=0.20           -> 23%
    # Bigger images do not help and even hurt slightly; the CONFIDENCE THRESHOLD
    # is the knob that matters. Worth knowing, because the detection gap is the
    # dominant failure mode of the whole system: on that flight YOLO returned
    # zero detections on 33% of AUTO frames, and each of those turns into a
    # target loss and then a hover.
    def __init__(self, weights="yolo11n.pt", imgsz=320, tracker="bytetrack.yaml",
                 conf=0.35, classes=None):
        self._model = YOLO(weights)      # heavy init hidden in the constructor
        self._imgsz = imgsz
        self._tracker = tracker
        self._conf = conf
        self._classes = classes
        # Map {class number: name}, e.g. {0: "person", 56: "chair"}. Knowledge of
        # the names lives here (the detector owns the model); everything
        # downstream gets a ready-made name and the renderer needs to know
        # nothing about YOLO.
        self._names = self._model.names

    def detect(self, frame) -> list[Detection]:
        results = self._model.track(
            frame, imgsz=self._imgsz, persist=True,
            tracker=self._tracker, verbose=False,
            conf=self._conf, classes=self._classes,
        )
        boxes = results[0].boxes
        # NOTE ON THIS EARLY RETURN. Dropping every box while the tracker has not
        # assigned ids yet means a confidently detected person can be discarded
        # for lack of a track id. Measured over a contiguous 901-frame stretch of
        # flight_20261002_185914 that costs 1% of frames -- real but minor, so
        # the simple behaviour stays. Do not reach for it to explain a large
        # detection gap; it is nowhere near big enough to be the cause.
        if boxes is None or boxes.id is None:   # nothing found / no ids yet
            return []

        # YOLO's tensors stop here: everything downstream sees Detection only.
        return [
            Detection(id = int(track_id), xyxy = tuple(xyxy), conf = conf, cls = int(cls),
                      name = self._names.get(int(cls), ""))
            for xyxy, track_id, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]
