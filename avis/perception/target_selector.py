# Вибір і супровід цілі за track id + пільговий період на тимчасову втрату.
# Звичайний клас без інтерфейсу (реалізація одна).


class TargetSelector:
    """Тримає, який відстежуваний об'єкт є ціллю, і чи він зараз видимий."""

    def __init__(self, lost_grace_frames=30):
        self._target_id = None                       # id цілі; None = не обрано
        self._lost_grace_frames = lost_grace_frames  # скільки кадрів прощаємо зникнення
        self._frames_since_seen = 0                  # лічильник кадрів без цілі

    # Доступ на ЧИТАННЯ id цілі, без можливості перезапису ззовні (інкапсуляція).
    @property
    def target_id(self):
        return self._target_id

    def select_at(self, x, y, detections):
        """Обрати новою ціллю ту рамку, що містить точку (x, y)."""
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:      # ланцюгове порівняння
                self._target_id = det.id
                self._frames_since_seen = 0
                return det
        return None

    def clear(self):
        self._target_id = None
        self._frames_since_seen = 0

    def resolve(self, detections):
        """Повертає (detection, lost). detection=None якщо ціль не обрано або
        пільгу вичерпано; lost=True — ціль обрано, але в цьому кадрі не видно."""
        if self._target_id is None:
            return None, False

        for det in detections:
            if det.id == self._target_id:
                self._frames_since_seen = 0
                return det, False

        self._frames_since_seen += 1
        if self._frames_since_seen > self._lost_grace_frames:
            self.clear()
            return None, False
        return None, True
