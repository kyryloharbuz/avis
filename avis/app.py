# App — "диригент" системи. Отримує всі залежності через конструктор
# (Dependency Injection) і не створює їх сам. Конкретні реалізації підставляє
# main.py (composition root).
#
# "Золота середина" видно й у типах параметрів: FrameSource та Renderer — це
# АБСТРАКЦІЇ (там підстановка реальна: Tello, тести), а решта — конкретні класи
# (одна реалізація, інтерфейс поки не потрібен). DI працює однаково для обох.

import time

import cv2  # App потрібен лише для констант/обробки вводу вікна (клавіші, миша)

# Імпортуємо через "фасади" підпакетів (їхні __init__.py), а не з глибоких шляхів.
from avis.control import ErrorCalculator, FollowController
from avis.perception import Detector, FrameSource, KalmanFilter, TargetSelector
from avis.view import Renderer

# Клавіші виходу. Esc (27) — надійний вихід у будь-якій розкладці й з CapsLock.
# Літери — best-effort: q/Q (англ./нім.), й/Й (укр./рос.), обидва регістри — CapsLock.
_QUIT_KEYS = {27, ord('q'), ord('Q'), ord('й'), ord('Й')}


class App:
    def __init__(
        self,
        source: FrameSource,            # абстракція (буде Tello)
        detector: Detector,             # конкретний клас
        selector: TargetSelector,       # конкретний клас
        tracker: KalmanFilter,          # конкретний клас
        error_calculator: ErrorCalculator,   # конкретний клас
        controller: FollowController,   # конкретний клас
        renderer: Renderer,             # абстракція (буде Headless для тестів)
    ):
        self._source = source
        self._detector = detector
        self._selector = selector
        self._tracker = tracker
        self._error_calculator = error_calculator
        self._controller = controller
        self._renderer = renderer

        self._renderer.set_mouse_callback(self._on_mouse)

        self._last_detections = []   # останні рамки (для обробника кліку)
        self._last_area = 0.0        # остання відома площа цілі (для дистанції)
        self._prev_time = None       # час попереднього кадру (для dt)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._selector.select_at(x, y, self._last_detections)

    def run(self):
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break

                # --- Крок 0: dt (скільки секунд минуло з минулого кадру) ---
                now = time.time()
                dt = 0.0 if self._prev_time is None else now - self._prev_time
                self._prev_time = now
                dt = max(1e-3, min(dt, 0.1))   # затиск: не 0 і не величезний стрибок

                # --- Крок 1: детекція + трекінг ---
                t0 = time.time()
                detections = self._detector.detect(frame)
                self._last_detections = detections
                fps = 1 / (time.time() - t0)

                h, w = frame.shape[:2]

                # --- Крок 2: де ціль + згладжування фільтром ---
                target, lost = self._selector.resolve(detections)

                center = None
                if target is not None:
                    # Ціль видно: згладжуємо центр, запам'ятовуємо площу.
                    # `*target.center` розкладає кортеж (cx, cy) на два аргументи.
                    center = self._tracker.update(*target.center, dt)
                    self._last_area = target.area
                elif lost and self._tracker.initialized:
                    # Ціль тимчасово зникла: лише передбачаємо за минулою швидкістю.
                    center = self._tracker.predict(dt)
                else:
                    # Цілі немає зовсім: скидаємо фільтр і регулятор.
                    self._tracker.reset()
                    self._controller.reset()

                # --- Крок 3: помилка → регулятор → команди ---
                if center is not None:
                    cx, cy = center
                    error = self._error_calculator.compute(cx, cy, self._last_area, w, h)
                    cmd = self._controller.update(error, dt)
                    print(
                        f"ex={error.error_x:+7.0f} ey={error.error_y:+7.0f} area={error.area:8.0f}"
                        f"   |   yaw={cmd.yaw:+6.1f} vert={cmd.vertical:+6.1f} fwd={cmd.forward:+6.1f}"
                    )

                # --- Крок 4: візуалізація ---
                self._renderer.draw(frame, detections, self._selector.target_id, fps, lost)

                # --- Крок 5: клавіатура / закриття вікна ---
                key = cv2.waitKey(1)
                if key != -1 and (key in _QUIT_KEYS or (key & 0xFF) in _QUIT_KEYS):
                    break
                if not self._renderer.is_open():
                    break
        finally:
            self._source.release()
            cv2.destroyAllWindows()
