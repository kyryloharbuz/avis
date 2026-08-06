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
    # align_full — до якого |norm_x| дозволено ПОВНИЙ хід уперед (0.15 = ціль
    #   майже по центру); align_stop — за яким |norm_x| рух уперед ПОВНІСТЮ
    #   заборонено (0.45 = ціль сильно збоку, спершу доверни).
    # deadzone — "зона спокою" навколо центру (частка півширини кадру).
    #   Якщо ціль ближче до центру, ніж deadzone — вважаємо, що вже добре, і
    #   НЕ рухаємось. Без цього дрон вічно смикався б навколо ідеального центру,
    #   ганяючись за шумом детектора на кілька пікселів.
    # retreat_factor — у СКІЛЬКИ разів повільніший рух НАЗАД (ціль наблизилась)
    #   проти руху вперед. 0.33 = утричі повільніше. Причина: відступати в
    #   приміщенні небезпечніше — дрон летить від цілі в бік, який "не бачить".
    def __init__(self, desired_area=60000.0, align_full=0.15, align_stop=0.45,
                 deadzone=0.05, area_deadzone=0.15, retreat_factor=0.33):
        self._desired_area = desired_area
        self._align_full = align_full
        self._align_stop = align_stop
        self._deadzone = deadzone
        self._area_deadzone = area_deadzone   # ±15% від бажаної площі = "дистанція ок"
        self._retreat_factor = retreat_factor
        # Коефіцієнти підняті ~вдвічі проти стартових — дрон реагував мляво.
        # kd теж піднято: різкіший kp без демпфера дає розхитування.
        # АГРЕСІЯ ПОВОРОТУ (наскільки різко реагує на зсув цілі від центру).
        # Історія: 0.38 → 0.57 → 0.855 (було заріздко) → 0.57 (÷1.5, поточне).
        self._pid_x = PID(kp=0.57, ki=0.0, kd=0.06)          # горизонталь → yaw
        self._pid_y = PID(kp=0.35, ki=0.0, kd=0.045)         # вертикаль → висота
        self._pid_area = PID(kp=0.0016, ki=0.0, kd=0.00012)  # дистанція → вперед/назад

    def reset(self):
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_area.reset()

    # trust_distance — чи МОЖНА зараз довіряти площі рамки (тобто дистанції).
    #   False під час PRED: свіжого виміру немає, площа застаріла. Керувати
    #   дистанцією за старим числом небезпечно — саме так дрон "летів уперед"
    #   на людину, вважаючи її далекою. У PRED керуємо лише напрямком.
    def update(self, error: ControlError, dt, trust_distance=True) -> Command:
        # Горизонталь центруємо ПОВОРОТОМ (yaw), як людина-оператор.
        yaw = self._pid_x.update(error.error_x, dt)
        # Вісь y дивиться ВНИЗ, тож інвертуємо знак, щоб + = вгору.
        vertical = -self._pid_y.update(error.error_y, dt)

        if not trust_distance:
            # Дистанцію не чіпаємо. PID площі теж скидаємо, щоб він не наздоганяв
            # накопиченим станом, коли ціль повернеться у поле зору.
            self._pid_area.reset()
            return Command(yaw=self._gate_yaw(yaw, error),
                           vertical=self._gate_vertical(vertical, error),
                           forward=0.0)

        # Помилка дистанції = бажана площа - поточна. Далеко → +, тобто вперед.
        forward = self._pid_area.update(self._desired_area - error.area, dt)

        # ── ТРИ ЗОНИ ДИСТАНЦІЇ + АСИМЕТРІЯ ─────────────────────────────────
        #   |area-desired| < area_deadzone → forward=0   SAFE — вис
        #   area < desired (ціль далеко)   → forward > 0  APPROACH — вперед
        #   area > desired (ціль близько)  → forward < 0  RETREAT — назад ×0.33
        if forward < 0:
            forward *= self._retreat_factor

        # Рух уперед дозволяємо ЛИШЕ тою мірою, якою вже дивимось на ціль:
        # якщо ціль збоку — спершу довертаємось, тоді їдемо (щоб не летіти повз).
        forward *= self._alignment(error.norm_x)

        # ЗОНИ СПОКОЮ: біля центру/потрібної відстані не смикаємось за шумом.
        if self._desired_area and \
                abs(self._desired_area - error.area) < self._area_deadzone * self._desired_area:
            forward = 0.0

        return Command(yaw=self._gate_yaw(yaw, error),
                       vertical=self._gate_vertical(vertical, error),
                       forward=forward)

    # Зони спокою винесені в методи, щоб застосовувались однаково і в
    # звичайному режимі, і коли дистанції не довіряємо.
    def _gate_yaw(self, yaw, error):
        return 0.0 if abs(error.norm_x) < self._deadzone else yaw

    def _gate_vertical(self, vertical, error):
        return 0.0 if abs(error.norm_y) < self._deadzone else vertical

    def _alignment(self, norm_x) -> float:
        """Коефіцієнт 0..1: наскільки ми "дивимось на ціль" і можемо їхати вперед."""
        offset = abs(norm_x)
        if offset <= self._align_full:
            return 1.0
        if offset >= self._align_stop:
            return 0.0
        # Лінійний спад між порогами.
        return (self._align_stop - offset) / (self._align_stop - self._align_full)


# ── Як налаштовувати (tuning) ──────────────────────────────────────────────
# 1. Почни з ki=0, kd=0. Піднімай kp, поки дрон впевнено не піде до цілі.
# 2. Розхитує ціль — додай трохи kd (демпфер).
# 3. Лишається стала недосяжність — додай зовсім трохи ki.
# 4. Дрона ще немає — команди друкуються, тож експериментувати безпечно.
