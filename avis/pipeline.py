# КОНВЕЄР ОБРОБКИ КАДРУ — чиста обробка без вводу/виводу.
#
# НАВІЩО ВИДІЛЕНО: раніше вся ця логіка жила всередині App.run() — метод на
# 150 рядків, що поєднував десяток обов'язків: таймінг, детекцію, клік,
# трекінг, керування, safety, статистику, запис, рендер і клавіатуру.
# Читати його доводилось цілком, щоб зрозуміти будь-яку частину.
#
# Тепер поділ чіткий:
#   TrackingPipeline — "що бачимо і куди летіти" (обчислення, без побічних дій)
#   App              — "звідки кадри, куди малювати, що натиснули" (оркестрація)
#
# Класи не знають одне про одного більше, ніж треба: конвеєр не має уявлення
# про вікна, клавіші й запис, а App не знає, як влаштований трекінг.

from dataclasses import dataclass, field

from avis.models import Command, ControlError

# Стани ведення цілі.
VIS = "VIS"     # ціль надійно видно — керуємо за виміром
PRED = "PRED"   # перекрита/зникла ненадовго — керуємо за прогнозом Калмана
LOST = "LOST"   # зникла надовго — цілі немає


@dataclass
class FrameResult:
    """Усе, що конвеєр дізнався про кадр. Незмінний знімок для App."""
    detections: list = field(default_factory=list)
    status: str = LOST
    center: tuple = None          # згладжена/передбачена позиція цілі
    predicted: tuple = None       # гіпотеза Калмана (для показу прицілу)
    error: ControlError = None
    command: Command = None       # чого хоче регулятор (ще не фільтровано safety)
    target_id: int = None
    off_frame: bool = False       # прогноз вийшов за межі кадру
    detect_seconds: float = 0.0
    # Кадр побитий втратою пакетів — на ньому сміття, а не сцена. Це НЕ те саме,
    # що "цілі немає": детекція на таких кадрах падає з 85% до 6%, тож
    # відсутність рамки тут нічого не доводить.
    frame_corrupt: bool = False

    @property
    def has_target(self) -> bool:
        """True ЛИШЕ коли ціль реально видно.

        Це важлива деталь: у стані PRED позиція існує, але вона — гіпотеза.
        Колись сюди підставляли "center is not None", і safety-логіка вважала,
        що ціль є, — дрон ганявся за привидом на повній швидкості."""
        return self.status == VIS

    @property
    def blind(self) -> bool:
        """Кадр не дає ЖОДНОЇ інформації про ціль: він побитий, і ми її не бачимо.

        Хто це читає, той не має права трактувати відсутність цілі як втрату —
        відсутність доказу не є доказом відсутності."""
        return self.frame_corrupt and self.status != VIS


class TrackingPipeline:
    """Кадр → детекції → ціль → згладжування → команда керування."""

    def __init__(self, detector, selector, tracker, error_calculator, controller,
                 on_event=None, quality=None):
        self._detector = detector
        self._selector = selector
        self._tracker = tracker
        self._error_calculator = error_calculator
        self._controller = controller
        # Контроль якості кадру. Не інтерфейс і не в композиційному корені:
        # реалізація одна, і конвеєр без неї працює (quality=False вимикає).
        if quality is None:
            from avis.perception.frame_quality import FrameQuality
            quality = FrameQuality()
        self._quality = quality or None
        # Куди повідомляти про події (зміна цілі тощо). Функція, а не print —
        # щоб конвеєр не залежав від способу виводу й мовчав у тестах.
        self._on_event = on_event or (lambda msg: None)

        self._last_box = None          # остання рамка цілі (для дистанції)
        self._last_area = 0.0
        self._active_target_id = None  # ловимо підміну цілі

    # ── Вибір цілі (делегуємо селектору) ───────────────────────────────
    def select_at(self, x, y, detections):
        return self._selector.select_at(x, y, detections)

    def select_target(self, det):
        return self._selector.select_target(det)

    @property
    def target_id(self):
        return self._selector.target_id

    # ── Головний крок ──────────────────────────────────────────────────
    def process(self, frame, dt, detections=None) -> FrameResult:
        """detections=None → детектуємо самі; інакше беремо готові (реплей)."""
        import time
        h, w = frame.shape[:2]

        corrupt = self._quality.is_corrupt(frame) if self._quality else False

        t0 = time.time()
        if detections is None:
            detections = self._detector.detect(frame)
        detect_seconds = time.time() - t0

        predicted, off_frame = self._predict(dt, w, h)
        # ЧАС ДЛЯ ПРОГНОЗУ І ЧАС ДЛЯ ВТРАТИ — це РІЗНІ речі, і тут вони
        # розходяться. Калман отримує справжній dt: час іде, ціль реально
        # рухається, прогноз має за нею встигати. А відлік втрати на побитому
        # кадрі СТОЇТЬ: ми не побачили ціль не тому, що її немає, а тому, що
        # кадр — сміття. Інакше пара секунд перешкод у Wi-Fi змушувала б дрон
        # оголосити ціль зниклою й почати висіти або сідати.
        target, lost = self._selector.resolve(detections, predicted,
                                              0.0 if corrupt else dt)

        result = FrameResult(detections=detections, predicted=predicted,
                             off_frame=off_frame, detect_seconds=detect_seconds,
                             target_id=self._selector.target_id,
                             frame_corrupt=corrupt)

        if target is not None:
            result.status = VIS
            result.center = self._on_visible(target)
        elif lost:
            result.status = PRED
            result.center = predicted
        else:
            result.status = LOST
            self._reset_tracking()

        if result.center is not None:
            cx, cy = result.center
            result.error = self._error_calculator.compute(
                cx, cy, self._last_area, w, h,
                box=self._last_box if result.status == VIS else None)
            # Дистанції довіряємо ЛИШЕ за свіжим виміром: у PRED рамка стара.
            result.command = self._controller.update(
                result.error, dt, trust_distance=(result.status == VIS))

        return result

    # ── Внутрішні кроки ────────────────────────────────────────────────
    def _predict(self, dt, w, h):
        """Прогноз Калмана, затиснутий у межі кадру.

        Затиск потрібен, бо модель сталої швидкості екстраполює НАЗОВНІ, коли
        ціль вийшла за кадр. Без нього регулятор ганявся б за точкою поза
        екраном, а перезахоплення шукало б ціль там, де рамок не буває."""
        if not self._tracker.initialized:
            return None, False
        predicted = self._tracker.predict(dt)
        if predicted is None:
            return None, False
        px, py = predicted
        cpx = min(max(px, 0.0), w - 1.0)
        cpy = min(max(py, 0.0), h - 1.0)
        return (cpx, cpy), (cpx != px or cpy != py)

    def _on_visible(self, target):
        """Ціль видно: ловимо підміну, коригуємо фільтр, запам'ятовуємо рамку."""
        if self._active_target_id is not None and target.id != self._active_target_id:
            # Перезахоплення на ІНШИЙ об'єкт: позиція стрибає на сотні пікселів.
            # Без скидання похідна PID дає спайк, а Калман читає стрибок як
            # шалену швидкість — звідси різкі ривки.
            self._on_event(f"ціль змінилась #{self._active_target_id} → "
                           f"#{target.id} ({target.name}) — скидаю фільтр і регулятор")
            self._tracker.reset()
            self._controller.reset()
        self._active_target_id = target.id
        self._last_area = target.area
        self._last_box = target.xyxy
        return self._tracker.correct(*target.center)

    def _reset_tracking(self):
        self._tracker.reset()
        self._controller.reset()
        self._active_target_id = None
