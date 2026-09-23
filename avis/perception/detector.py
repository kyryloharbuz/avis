# Детектор об'єктів на YOLO + трекер ByteTrack.
#
# Тут ІНТЕРФЕЙСУ НЕМАЄ (золота середина): реалізація одна й іншої не
# передбачається. Клас має чисте загальне ім'я `Detector`. Якщо колись
# з'явиться друга реалізація — виділимо інтерфейс за 2 хвилини (додамо ABC
# і перейменуємо цей клас на YoloDetector).

from ultralytics import YOLO

from avis.models import Detection  # значення-об'єкт, який повертаємо


class Detector:
    # conf — мінімальна впевненість. Дефолт ultralytics 0.25 надто низький: у
    #   конвеєр летіли слабкі детекції, і саме вони давали плутанину
    #   "людина ↔ стілець". 0.5 відсікає сміття.
    # classes — які класи COCO взагалі детектувати. None = усі 80.
    #   [0] = лише люди — найнадійніший режим для follow-me: стільці, дивани й
    #   рослини тоді фізично не можуть стати ціллю.
    # conf=0.35, а не 0.5: заміряно на реальному записі (120 кадрів) — нижчий
    #   поріг дає +2.5% детекцій БЕЗ втрати якості (медіана впевненості 0.81,
    #   тобто слабких хибних спрацювань і так майже немає).
    # imgsz лишається 320: той самий замір показав, що 480/640 дають лише
    #   +5% детекцій ціною ВДВІЧІ більшого часу — межа не в роздільності,
    #   а в самому відео (втрата пакетів, розмиття).
    def __init__(self, weights="yolo11n.pt", imgsz=320, tracker="bytetrack.yaml",
                 conf=0.35, classes=None):
        self._model = YOLO(weights)      # важка ініціалізація схована в конструкторі
        self._imgsz = imgsz
        self._tracker = tracker
        self._conf = conf
        self._classes = classes
        # Словник {номер_класу: назва}, напр. {0: "person", 56: "chair"}.
        # Знання назв живе саме тут (детектор володіє моделлю) — далі по системі
        # йде вже готова назва, і рендеру не треба нічого знати про YOLO.
        self._names = self._model.names

    def detect(self, frame) -> list[Detection]:
        results = self._model.track(
            frame, imgsz=self._imgsz, persist=True,
            tracker=self._tracker, verbose=False,
            conf=self._conf, classes=self._classes,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:   # нікого не знайдено / ще нема id
            return []

        # "Брудні" тензори YOLO перетворюємо на охайні Detection — деталі
        # бібліотеки не витікають назовні (інкапсуляція).
        return [
            Detection(id = int(track_id), xyxy = tuple(xyxy), conf = conf, cls = int(cls),
                      name = self._names.get(int(cls), ""))
            for xyxy, track_id, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]
