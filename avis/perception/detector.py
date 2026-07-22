# Детектор об'єктів на YOLO + трекер ByteTrack.
#
# Тут ІНТЕРФЕЙСУ НЕМАЄ (золота середина): реалізація одна й іншої не
# передбачається. Клас має чисте загальне ім'я `Detector`. Якщо колись
# з'явиться друга реалізація — виділимо інтерфейс за 2 хвилини (додамо ABC
# і перейменуємо цей клас на YoloDetector).

from ultralytics import YOLO

from avis.models import Detection  # значення-об'єкт, який повертаємо


class Detector:
    def __init__(self, weights="yolo11n.pt", imgsz=320, tracker="bytetrack.yaml"):
        self._model = YOLO(weights)      # важка ініціалізація схована в конструкторі
        self._imgsz = imgsz
        self._tracker = tracker

    def detect(self, frame) -> list[Detection]:
        results = self._model.track(
            frame, imgsz=self._imgsz, persist=True,
            tracker=self._tracker, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:   # нікого не знайдено / ще нема id
            return []

        # "Брудні" тензори YOLO перетворюємо на охайні Detection — деталі
        # бібліотеки не витікають назовні (інкапсуляція).
        return [
            Detection(id = int(track_id), xyxy = tuple(xyxy), conf = conf, cls = int(cls))
            for xyxy, track_id, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]
