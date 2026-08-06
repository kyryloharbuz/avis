# Вибір і "пам'ять" цілі: тримає, яку рамку переслідуємо, і не губить її при
# частковій оклюзії (зайшла за шафу) чи короткому повному зникненні.
#
# Дві механіки:
#  1) ВИДИМІСТЬ. Перекрита ціль дає меншу рамку → % видимості = площа/еталон.
#     Якщо < min_visibility, центр рамки оманливий → віддаємо керування прогнозу.
#  2) ПАМ'ЯТЬ + ПЕРЕЗАХОПЛЕННЯ. Лічильник у секундах прощає коротку відсутність;
#     нову рамку того ж класу/розміру біля прогнозу приймаємо як ту саму ціль.


def _point_box_distance(x, y, xyxy):
    """Відстань від точки (x, y) до прямокутника; 0, якщо точка всередині."""
    x1, y1, x2, y2 = xyxy
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


class TargetSelector:
    """Тримає ціль і не губить її при оклюзії/короткому зникненні."""

    def __init__(
        self,
        lost_grace_seconds=3.0,       # СЕКУНД прощаємо повну відсутність
        min_visibility=0.18,          # < цього → ціль вважаємо перекритою
        occlusion_debounce=3,         # стільки КАДРІВ поспіль має триматись
                                      #   просідання, перш ніж повіримо в оклюзію
        reference_smoothing=0.1,      # EMA еталонної ПЛОЩІ
        reacquire_radius=120.0,       # базовий радіус (px) перезахоплення
        reacquire_growth=80.0,        # +px до радіуса за секунду відсутності
        reacquire_radius_max=220.0,   # стеля радіуса (було 400 — пів кадру,
                                      #   через це дрон хапав сторонні об'єкти)
    ):
        self._lost_grace_seconds = lost_grace_seconds
        self._min_visibility = min_visibility
        self._occlusion_debounce = occlusion_debounce
        self._reference_smoothing = reference_smoothing
        self._reacquire_radius = reacquire_radius
        self._reacquire_growth = reacquire_growth
        self._reacquire_radius_max = reacquire_radius_max

        self._target_id = None        # id поточної цілі; None = не обрано
        self._target_cls = None       # клас цілі (щоб не перечепитись на інший об'єкт)
        self._reference_area = None    # "повна" площа цілі (еталон видимості)
        self._time_since_seen = 0.0    # секунд від останнього надійного бачення
        self._occluded_frames = 0      # поспіль кадрів із просілою площею

    @property
    def target_id(self):
        return self._target_id

    def select_at(self, x, y, detections, tolerance=70.0):
        """Обрати ціль за кліком (x, y). Прощає неточність: якщо не влучив у
        рамку, беремо найближчу в межах tolerance пікселів."""
        inside = [d for d in detections
                  if d.xyxy[0] <= x <= d.xyxy[2] and d.xyxy[1] <= y <= d.xyxy[3]]
        if inside:
            # Якщо клік потрапив у кілька рамок одразу (людина стоїть біля
            # стільця, рамки перекриваються) — беремо ту, чий ЦЕНТР найближчий
            # до кліку. Раніше бралася НАЙМЕНША за площею, і клік по людині
            # поруч зі стільцем міг обрати стілець.
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
        """Прямо призначити ціллю конкретну рамку (клік або відтворення реплею)."""
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

    # % видимості = поточна площа / еталон. Раптове падіння (оклюзія) дає малий
    # відсоток, а повільна зміна від віддалення встигає всмоктатись в еталон (EMA).
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

    # Знайти ту саму ціль під НОВИМ id — найближчу рамку того ж класу/розміру
    # біля передбаченої Калманом позиції.
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
        """Повертає (detection, lost):
          (det,  False) — ціль надійно видно / перезахоплено;
          (None, True)  — оклюзія / коротка втрата: керуй за прогнозом Калмана;
          (None, False) — ціль зникла надовго або не обрана."""
        if self._target_id is None:
            return None, False

        current = next((d for d in detections if d.id == self._target_id), None)

        # Знайшли за id → перевіряємо ВИДИМІСТЬ.
        if current is not None:
            # КЛАС ЦІЛІ НЕ ПЕРЕЗАПИСУЄМО. Раніше тут стояло
            #   self._target_cls = current.cls
            # і це був баг: ByteTrack тримає той самий track id, а YOLO на один
            # кадр може класифікувати людину як стілець — і клас цілі НАЗАВЖДИ
            # ставав "стілець". Далі перезахоплення шукало вже стільці. Клас
            # фіксується один раз у select_target() і більше не змінюється.
            if self._visibility(current) >= self._min_visibility:
                self._time_since_seen = 0.0
                self._occluded_frames = 0
                self._update_reference(current.area)
                return current, False                 # надійно видно

            # ДЕБАУНС ОКЛЮЗІЇ. Один кадр із просілою площею — це найчастіше
            # мерехтіння детектора, а не реальна оклюзія. Оголошуємо втрату лише
            # коли просідання ТРИМАЄТЬСЯ кілька кадрів поспіль. Без цього
            # виникали хибні PRED (особливо при швидкій зміні дистанції), а в
            # PRED керування йшло за прогнозом зі старою площею.
            self._occluded_frames += 1
            if self._occluded_frames < self._occlusion_debounce:
                self._time_since_seen = 0.0
                return current, False                 # ще віримо рамці
            return self._tick_lost(dt)                 # стійка оклюзія

        # Не знайшли за id → геометричне перезахоплення біля прогнозу.
        if predicted_center is not None:
            candidate = self._try_reacquire(detections, predicted_center)
            if candidate is not None:
                # id оновлюємо (трекер дав новий), а КЛАС — ні: він зафіксований
                # при виборі цілі, і кандидат уже пройшов перевірку за класом.
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
