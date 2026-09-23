# ДИСТАНЦІЯ — вісь, яка й була джерелом головного бага.
#
# ЩО БУЛО НЕ ТАК: існували ДВІ незалежні уставки однієї фізичної величини —
# бажана ПЛОЩА рамки (94000 px² ≈ 1.44 м) і бажана ВІДСТАНЬ (1.7 м). Вони не
# збігались, і саме перемикання між ними ("є калібрування / немає") мовчки
# зсувало ціль на 26 см. Замір показав: рух уперед падав із 26% до 5%.
#
# ЯК ЗАРАЗ: одна уставка — В МЕТРАХ. Гілки "через площу" більше немає взагалі,
# бо DistanceEstimator віддає метри завжди (без калібрування — за номінальною
# константою). Один шлях коду, одне джерело істини.
#
# ЧОМУ МЕТРИ, А НЕ ПЛОЩА: площа ~ 1/відстань², тобто від 2 м до 1 м вона
# зростає ВЧЕТВЕРО, а відстань лише вдвічі. Через цю нелінійність дрон
# зблизька сахався ривком, а здалеку ледве повз. У метрах реакція лінійна.

from collections import deque

from avis.control.follow.axis import Axis


class DistanceAxis(Axis):
    """Тримає задану відстань до цілі. Працює ВИКЛЮЧНО в метрах."""

    def __init__(self, cfg):
        super().__init__(kp=cfg.distance_kp, kd=cfg.distance_kd,
                         deadzone=cfg.distance_deadzone_m)
        self._target = cfg.distance_target_m
        self._retreat_factor = cfg.retreat_factor
        self._prev_distance = None
        self._rate = 0.0             # м/с, потрібна вертикалі для заморозки

        # ЗАПОБІЖНИК ЗА ЧАСТКОЮ КАДРУ. Оцінка дистанції залежить від
        # калібрування й ПОЗИ: коли людина повертається боком, плечі в кадрі
        # вужчають, і система читає це як "ціль віддалилась" — та летить на неї.
        # Заміряно: ширина в 1.8 раза менш стабільна за зріст (16% проти 9%).
        # Частка кадру таких вад не має — рамка на півекрана означає "близько"
        # за будь-якої пози й без жодного калібрування.
        self._fill_stop = cfg.frame_fill_stop        # вище — рух уперед заборонено
        self._fill_retreat = cfg.frame_fill_retreat  # вище — примусовий відступ
        self._retreat_cmd = cfg.frame_fill_retreat_command

        # МЕДІАННИЙ ФІЛЬТР дистанції. Саме медіана (а не середнє) потрібна
        # тому, що поворот тіла дає ВИКИДИ, а не рівномірний шум: одиночний
        # хибний вимір медіана відкидає, і дрон не смикається на нього.
        self._window = deque(maxlen=cfg.distance_median_frames)

    @property
    def rate(self):
        """Швидкість зміни дистанції (м/с). Вертикаль використовує її, щоб
        відрізнити 'ціль присіла' від 'дрон підʼїхав'."""
        return self._rate

    def reset(self):
        super().reset()
        self._prev_distance = None
        self._rate = 0.0
        self._window.clear()

    def smooth(self, distance_m):
        """Медіана останніх вимірів — гасить викиди від зміни пози."""
        if distance_m is None:
            return None
        self._window.append(distance_m)
        ordered = sorted(self._window)
        return ordered[len(ordered) // 2]

    def track_rate(self, distance_m, dt):
        """Оновити оцінку швидкості зміни дистанції. Викликати ЩОКАДРУ,
        навіть коли команду не рахуємо — інакше похідна брехатиме."""
        if distance_m is None or dt <= 0:
            self._rate = 0.0
            return
        if self._prev_distance is not None:
            raw = (distance_m - self._prev_distance) / dt
            self._rate = 0.7 * self._rate + 0.3 * raw   # згладжуємо: 5-7 FPS шумні
        self._prev_distance = distance_m

    def command(self, error, dt):
        """Помилка = поточна відстань - бажана. Далеко → +, тобто вперед."""
        # ЗАПОБІЖНИК ПЕРШИЙ, до будь-яких обчислень. Якщо рамка займає надто
        # багато кадру — ми близько НЕЗАЛЕЖНО від того, що показує оцінка
        # дистанції. Це і рятує від головної вади вимірювання шириною.
        if error.frame_fill >= self._fill_retreat:
            self._pid.reset()
            # Величина ФІКСОВАНА і завідомо вища за поріг чутливості дрона:
            # запобіжник, який дрон не відпрацьовує, — гірший за відсутній.
            return -abs(self._retreat_cmd)

        distance = self.smooth(error.distance_m)
        if distance is None:
            return self.hold()       # без виміру дистанцією не керуємо

        out = self.update(distance - self._target, dt)

        # ВІДСТУП повільніший за наступ: позаду дрон нічого не бачить.
        if out < 0:
            out *= self._retreat_factor

        # Рух УПЕРЕД забороняємо у двох випадках:
        #   • рамка вже завелика (близько за прямим сигналом);
        #   • дистанцію виміряти неможливо — отже ми, найімовірніше, впритул.
        if error.frame_fill >= self._fill_stop or error.box_clipped:
            return min(out, 0.0)
        return out
