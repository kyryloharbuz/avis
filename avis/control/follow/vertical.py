# ВЕРТИКАЛЬ — найхитріша вісь, і ось чому.
#
# ПРОБЛЕМА 1 (заміряно на 10 польотах): центр рамки людини систематично стоїть
# ВИЩЕ центру кадру (y≈230 проти 360). Якщо ціль — центр кадру, дрон отримує
# команду "вгору" БЕЗПЕРЕРВНО й повзе до стелі.
#   → Рішення: цілимось не в центр кадру, а у ВЛАСНИЙ ЗВИЧНИЙ РІВЕНЬ ЦІЛІ,
#     який вивчаємо повільною ковзною середньою. Постійний зсув всмоктується
#     і перестає давати команду; швидка зміна (присів/встав) — ні.
#
# ПРОБЛЕМА 2 (теж заміряно): коли дрон НАБЛИЖАЄТЬСЯ, центр рамки їде вниз на
# ~56px САМ ПО СОБІ — це чиста геометрія, а не рух цілі. Адаптивний рівень
# читав це як "присіла" і знижував дрон.
#   → Рішення: поки ДИСТАНЦІЯ ЗМІНЮЄТЬСЯ швидше за поріг — вісь ЗАМОРОЖУЄМО.
#     Присідання ж відбувається БЕЗ зміни дистанції, тому воно проходить.

from avis.control.follow.axis import Axis


class VerticalAxis(Axis):
    """Тримає ціль на її звичному рівні; реагує на присідання, не на підліт."""

    def __init__(self, cfg):
        super().__init__(kp=cfg.vertical_kp, kd=cfg.vertical_kd,
                         deadzone=cfg.vertical_deadzone_px)
        self._enabled = cfg.vertical_enabled
        self._tau = cfg.vertical_adapt_tau_s
        self._freeze_speed = cfg.vertical_freeze_speed_mps
        self._reference = None       # звичний рівень цілі (px помилки від центру)

    def reset(self):
        super().reset()
        self._reference = None       # рівень належить КОНКРЕТНІЙ цілі

    def frozen(self, error, distance_rate) -> bool:
        """Чи можна ЗАРАЗ довіряти вертикальній геометрії."""
        if error.box_clipped:
            return True              # рамку обрізало краєм — центр недостовірний
        return abs(distance_rate) > self._freeze_speed

    def command(self, error, dt, distance_rate):
        if not self._enabled:
            return 0.0          # висоту тримає сам Tello (див. коментар у config)
        if self.frozen(error, distance_rate):
            self._pid.reset()        # не вчимось і не рухаємось на брехливих даних
            return 0.0

        if self._reference is None:
            self._reference = error.error_y
            return 0.0

        deviation = error.error_y - self._reference
        # Повільно підтягуємо рівень до поточного. Вісь y дивиться ВНИЗ, тож
        # інвертуємо знак: ціль нижче звичного → летимо ВНИЗ.
        a = min(1.0, dt / self._tau)
        self._reference = (1 - a) * self._reference + a * error.error_y
        return -self.update(deviation, dt)
