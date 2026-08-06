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
    # Коефіцієнти PID — ПАРАМЕТРИ, а не константи всередині. Це потрібно саме
    # для тюнінгу на реплеї: можна прогнати той самий запис із десятком різних
    # kp у циклі й порівняти числа, не редагуючи файл щоразу.
    def __init__(self, desired_area=94000.0, align_full=0.35, align_stop=0.80,
                 deadzone=0.05, area_deadzone=0.15, retreat_factor=0.33,
                 kp_yaw=0.40, kd_yaw=0.10,
                 kp_vertical=0.35, kd_vertical=0.045,
                 kp_area=0.0016, kd_area=0.00012,
                 vertical_adapt_tau=8.0, vertical_deadzone_px=45.0):
        self._desired_area = desired_area
        self._align_full = align_full
        self._align_stop = align_stop
        self._deadzone = deadzone
        self._area_deadzone = area_deadzone   # ±15% від бажаної площі = "дистанція ок"
        self._retreat_factor = retreat_factor

        # ── АДАПТИВНА ВЕРТИКАЛЬ ────────────────────────────────────────────
        # Проблема, яку це лікує (виміряна на 10 польотах): центр рамки людини
        # СИСТЕМАТИЧНО стоїть вище центру кадру (y≈230 проти 360, тобто ≈-130px)
        # на будь-якій дистанції. Через це дрон отримував команду "вгору"
        # БЕЗПЕРЕРВНО й повільно ліз до стелі.
        #
        # Рішення: цілимось не в центр кадру, а у ВЛАСНИЙ ЗВИЧНИЙ РІВЕНЬ ЦІЛІ.
        # Повільна ковзна середня (tau ≈ 8 с) вбирає постійний зсув, тож він
        # більше не дає команди. А ШВИДКА зміна — присів або встав — за 8 секунд
        # всмоктатись не встигає, дає велике відхилення й дрон іде вниз/вгору.
        # Тобто: постійний нахил ігноруємо, реальний рух цілі відпрацьовуємо.
        self._vertical_adapt_tau = vertical_adapt_tau
        self._vertical_deadzone_px = vertical_deadzone_px
        self._vertical_ref = None      # звичний рівень цілі (у пікселях помилки)
        self._vertical_dev = 0.0       # поточне відхилення від цього рівня
        # kp_yaw — АГРЕСІЯ ПОВОРОТУ. Знижено 0.57 → 0.40, а демпфер kd
        #   піднято 0.06 → 0.10. Причина з даних польоту: поворот був у
        #   САТУРАЦІЇ 49% часу — тобто крутив на максимумі майже безперервно,
        #   і це відчувалось як кидання з боку в бік. Менший kp + більший
        #   демпфер дають плавніше ведення ціною трохи повільнішого доводу.
        self._pid_x = PID(kp=kp_yaw, ki=0.0, kd=kd_yaw)            # горизонталь → yaw
        self._pid_y = PID(kp=kp_vertical, ki=0.0, kd=kd_vertical)  # вертикаль → висота
        self._pid_area = PID(kp=kp_area, ki=0.0, kd=kd_area)       # дистанція → вперед/назад

    def reset(self):
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_area.reset()
        # Звичний рівень — властивість КОНКРЕТНОЇ цілі, тож при зміні цілі
        # (або втраті) його треба вчити наново.
        self._vertical_ref = None
        self._vertical_dev = 0.0

    def _vertical_deviation(self, error, dt):
        """Наскільки ціль зараз вище/нижче за СВІЙ звичний рівень, у пікселях.
        0 означає "ціль там, де зазвичай" — тобто постійний зсув ігнорується."""
        if self._vertical_ref is None:
            self._vertical_ref = error.error_y      # перше бачення = еталон
            self._vertical_dev = 0.0
            return 0.0
        self._vertical_dev = error.error_y - self._vertical_ref
        # Повільно підтягуємо еталон до поточного рівня. Що більше tau, то
        # довше "пам'ятає" і то впевненіше реагує на присідання.
        a = min(1.0, dt / self._vertical_adapt_tau)
        self._vertical_ref = (1 - a) * self._vertical_ref + a * error.error_y
        return self._vertical_dev

    # trust_distance — чи МОЖНА зараз довіряти площі рамки (тобто дистанції).
    #   False під час PRED: свіжого виміру немає, площа застаріла. Керувати
    #   дистанцією за старим числом небезпечно — саме так дрон "летів уперед"
    #   на людину, вважаючи її далекою. У PRED керуємо лише напрямком.
    def update(self, error: ControlError, dt, trust_distance=True) -> Command:
        # Горизонталь центруємо ПОВОРОТОМ (yaw), як людина-оператор.
        yaw = self._pid_x.update(error.error_x, dt)
        # ВЕРТИКАЛЬ: керуємо не за помилкою від центру кадру, а за ВІДХИЛЕННЯМ
        # цілі від її ж звичного рівня (див. коментар у конструкторі).
        # Вісь y дивиться ВНИЗ, тож інвертуємо знак, щоб + = вгору.
        vertical = -self._pid_y.update(self._vertical_deviation(error, dt), dt)

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
        """М'ЯКА зона спокою замість жорсткої.

        Було: усередині зони — рівно 0, а на крок за неї — одразу повна
        команда. Це "сходинка": відходиш убік, і дрон різко смикається в
        поворот, хоч зсув ще малий.

        Стало: біля межі зони команда починається з НУЛЯ і росте плавно —
        ривка немає, а спокій біля центру зберігається."""
        offset = abs(error.norm_x)
        if offset < self._deadzone:
            return 0.0
        # 0 на самій межі зони → 1 при великому зсуві.
        ramp = (offset - self._deadzone) / max(1.0 - self._deadzone, 1e-6)
        return yaw * min(ramp / max(offset, 1e-6), 1.0)

    def _gate_vertical(self, vertical, error):
        """Зона спокою вертикалі — по ВІДХИЛЕННЮ від звичного рівня цілі,
        у ПІКСЕЛЯХ. Поріг великий (~45px) навмисно: дрібне тремтіння рамки не
        має рухати дрон, а присідання людини зсуває центр на 100-150px —
        тобто впевнено перевищує поріг і спрацьовує."""
        return 0.0 if abs(self._vertical_dev) < self._vertical_deadzone_px else vertical

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
