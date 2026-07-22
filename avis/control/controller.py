# Регулятор: перетворює помилку на команду руху. Всередині — три PID-и.
#
# PID у трьох словах (кермуємо як водій):
#   P — реагуй пропорційно поточній помилці (далеко → сильніше).
#   I — накопичуй малу сталу помилку в часі й дотискай її до нуля.
#   D — дивись, ЯК ШВИДКО помилка змінюється, і гальмуй заздалегідь (демпфер).
# Вихід = P + I + D.

from avis.models import Command, ControlError


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


class FollowController:
    """КОМПОНУЄ три PID-и (по осі) і зводить їх у Command."""

    # desired_area — бажана площа рамки ("тримай дистанцію"): ближча ціль → назад,
    #   дальша → вперед. Підбирається під камеру.
    def __init__(self, desired_area=60000.0):
        self._desired_area = desired_area
        # Коефіцієнти — СТАРТОВІ, їх треба тюнити (див. підказку внизу).
        self._pid_x = PID(kp=0.20, ki=0.0, kd=0.02)          # горизонталь → yaw
        self._pid_y = PID(kp=0.20, ki=0.0, kd=0.02)          # вертикаль → висота
        self._pid_area = PID(kp=0.0015, ki=0.0, kd=0.0001)   # дистанція → вперед/назад

    def reset(self):
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_area.reset()

    def update(self, error: ControlError, dt) -> Command:
        yaw = self._pid_x.update(error.error_x, dt)          # правіше → повертаємось праворуч
        # Вісь y дивиться ВНИЗ, тож інвертуємо знак, щоб + = вгору.
        vertical = -self._pid_y.update(error.error_y, dt)
        # Помилка дистанції = бажана площа - поточна. Далеко → +, тобто вперед.
        forward = self._pid_area.update(self._desired_area - error.area, dt)
        return Command(yaw=yaw, vertical=vertical, forward=forward)


# ── Як налаштовувати (tuning) ──────────────────────────────────────────────
# 1. Почни з ki=0, kd=0. Піднімай kp, поки дрон впевнено не піде до цілі.
# 2. Розхитує ціль — додай трохи kd (демпфер).
# 3. Лишається стала недосяжність — додай зовсім трохи ki.
# 4. Дрона ще немає — команди друкуються, тож експериментувати безпечно.
