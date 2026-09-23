# PID — базовий будівельний блок регуляторів.
#
# Раніше в цьому файлі жив ще й FollowController на 250 рядків із пʼятьма
# відповідальностями. Його розібрано на пакет avis/control/follow/, де кожна
# вісь — окремий клас зі своїми тестами. Тут лишилась сама математика PID.
#
# PID у трьох словах (кермуємо як водій):
#   P — реагуй пропорційно поточній помилці (далеко → сильніше).
#   I — накопичуй малу сталу помилку в часі й дотискай її до нуля.
#   D — дивись, ЯК ШВИДКО помилка змінюється, і гальмуй заздалегідь (демпфер).
# Вихід = P + I + D.


def _clamp(value, lo, hi):                  # "затискачка" в межі (нема вбудованого)
    return max(lo, min(value, hi))


# PID — ВНУТРІШНІЙ будівельний блок (не публічний API). Тому його немає в
# __init__.py пакета: ним користується лише FollowController (композиція).
class PID:
    """PID-регулятор для ОДНІЄЇ величини."""

    def __init__(self, kp, ki, kd, output_limit=100.0, integral_limit=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._output_limit = output_limit       # обмеження виходу (діапазон RC Tello)
        self._integral_limit = integral_limit   # захист від "integral windup"
        self._integral = 0.0                     # накопичена сума (I)
        self._prev_error = None                  # попередня помилка (для D)

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def update(self, error, dt):
        if dt <= 0:                              # захист від ділення на 0 у складовій D
            return 0.0

        p = self.kp * error                      # --- P ---

        self._integral += error * dt             # --- I ---
        self._integral = _clamp(self._integral, -self._integral_limit, self._integral_limit)
        i = self.ki * self._integral

        if self._prev_error is None:             # --- D ---
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        d = self.kd * derivative
        self._prev_error = error

        return _clamp(p + i + d, -self._output_limit, self._output_limit)
