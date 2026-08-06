# SAFETY-ЛОГІКА автономного польоту (крок 6 roadmap).
#
# Це найважливіший клас у проєкті з точки зору безпеки. PID рахує, КУДИ летіти,
# а FlightSupervisor вирішує, чи МОЖНА зараз туди летіти взагалі.
#
# Головні принципи, закладені сюди:
#   1. АВТОНОМНІСТЬ ВМИКАЄ ЛЮДИНА. Після злету дрон просто висить. Він не
#      починає переслідування, доки оператор явно не натисне "G" (arm).
#      Дрон, що сам рушає одразу після злету, — найкращий спосіб щось розбити.
#   2. ВТРАТИЛИ ЦІЛЬ → ВИСИМО, а не летимо наосліп останньою командою.
#   3. НЕ ЗНАЙШЛИ ДОВГО → САДИМО САМІ. Дрон не має висіти вічно.
#   4. КОМАНДИ ПРИГЛУШЕНІ. PID видає -100..100, ми масштабуємо до безпечних
#      значень: перший автономний політ на повній швидкості — погана ідея.
#   5. РОЗРЯДИВСЯ → САДИМО.

import threading

from avis.models import Command

# Стани польоту.
GROUNDED = "GROUNDED"   # на землі, мотори вимкнені
HOVER = "HOVER"         # у повітрі, висить, автономність ВИМКНЕНА
AUTO = "AUTO"           # у повітрі, керує PID
BUSY = "BUSY"           # виконується зліт/посадка (команда ще не відповіла)


class FlightSupervisor:
    """Стежить за станом польоту й фільтрує команди PID перед моторами."""

    def __init__(
        self,
        drone,
        max_speed=70,              # МЕЖА швидкості повороту й вертикалі
        max_forward=30,            # МЕЖА швидкості руху ВПЕРЕД (ризикова вісь)
        max_backward=15,           # МЕЖА швидкості руху НАЗАД — удвічі нижча за вперед.
                                   #   Коли людина йде на дрон, площа рамки росте
                                   #   різко, і регулятор просив повний хід назад:
                                   #   дрон сахався надто швидко. Відступ має бути
                                   #   спокійним — позаду він нічого не бачить.
        min_speed=12,              # нижче цього Tello просто ігнорує команду
        slew_per_second=90.0,      # ПЛАВНІСТЬ: на скільки одиниць за секунду
                                   #   команді дозволено НАРОСТАТИ. Гальмування
                                   #   (до нуля) НЕ обмежене — зупинка миттєва.
                                   #   Лікує різкий ривок повороту: замість
                                   #   стрибка 0→30 команда доходить за ~0.3 с.
        hover_after_lost=0.5,      # с без цілі → перестати рухатись, висіти
        land_after_lost=8.0,       # с без цілі → сідати самому (лише ПІСЛЯ 1-го захоплення)
        settle_after_takeoff=5.0,  # с ЗАВИСУ після зльоту: Tello стабілізується
                                   # і встигаєш зачепити ціль, перш ніж дрон рушить
        min_battery=15,            # % → примусова посадка
        enable_forward=False,      # рух ВПЕРЕД/НАЗАД: за замовчуванням ВИМКНЕНО
        start_armed=False,         # одразу в AUTO (для РЕПЛЕЮ з NullDrone, без злету)
        min_height_cm=50,          # нижче — блокуємо зниження (захист підлоги)
        max_height_cm=180,         # вище — блокуємо набір висоти (захист стелі)
        height_margin_cm=30,       # зона плавного гальмування біля меж
        altitude_limits=False,     # ВИМКНЕНО: висотомір Tello шумить/дрейфує і
                                   # може блокувати вертикаль. Увімкнеш, коли
                                   # переконаєшся, що стеження стабільне.
    ):
        self._drone = drone
        self._max_speed = max_speed
        self._max_forward = max_forward
        self._max_backward = max_backward
        self._min_speed = min_speed
        self._slew = slew_per_second
        self._prev_cmd = Command(0.0, 0.0, 0.0)
        self._min_height = min_height_cm
        self._max_height = max_height_cm
        self._height_margin = height_margin_cm
        self._altitude_limits = altitude_limits
        self._hover_after_lost = hover_after_lost
        self._land_after_lost = land_after_lost
        self._min_battery = min_battery
        self._enable_forward = enable_forward

        # start_armed=True лише для реплею: там немає реального дрона й злету,
        # але ми хочемо бачити повний ланцюг команд аж до "моторів".
        self._settle_after_takeoff = settle_after_takeoff

        self._state = AUTO if start_armed else GROUNDED
        self._time_without_target = 0.0
        self._battery_checked_at = 0.0
        self._settle = 0.0            # лічильник "відстоювання" після зльоту
        self._ever_acquired = False   # чи хоч раз захопили ціль у цьому польоті
        self._last_battery = None     # кеш заряду (щоб писати в лог без зайвих запитів)
        self._busy = False            # виконується зліт/посадка у фоновому потоці
        self._arm_requested = False   # G натиснули ще під час зльоту

    @property
    def state(self):
        # Поки триває зліт/посадка показуємо BUSY — і оператору видно, що команда
        # в дорозі, і цикл знає, що слати RC зараз не можна.
        return BUSY if self._busy else self._state

    @property
    def armed(self):
        return self._state == AUTO

    @property
    def battery(self):
        """Останній ЗЧИТАНИЙ заряд (кеш). Не робить нового запиту до дрона —
        значення оновлюється у update() раз на ~2 с і при зльоті."""
        return self._last_battery

    # ── Команди оператора ──────────────────────────────────────────────
    #
    # ЧОМУ ЗЛІТ І ПОСАДКА — У ФОНОВОМУ ПОТОЦІ:
    # djitellopy виконує takeoff() СИНХРОННО й чекає відповіді дрона аж до
    # TAKEOFF_TIMEOUT = 20 секунд. Якщо викликати це просто з циклу кадрів,
    # весь цикл СТОЇТЬ: кадри не читаються, вікно не перемальовується — і
    # виглядає, ніби відео зависло на 15+ секунд. Тому команду виконуємо в
    # окремому потоці, а цикл тим часом малює далі.
    # Поки команда в дорозі, стан = BUSY, і в мотори НІЧОГО не шлемо (щоб RC не
    # конфліктував із виконанням зльоту).
    def _run_async(self, label, action, on_done):
        def worker():
            try:
                action()
                on_done()
            except Exception as e:
                print(f"[flight] {label} не вдалось: {type(e).__name__}: {e}")
                self._state = GROUNDED
            finally:
                self._busy = False
        self._busy = True
        threading.Thread(target=worker, daemon=True).start()

    def takeoff(self):
        if self._state != GROUNDED or self._busy:
            return
        battery = self._drone.battery()
        self._last_battery = battery
        if battery < 30:
            print(f"[flight] ВІДМОВА: заряд {battery}% (треба >30%)")
            return
        print(f"[flight] злітаю (заряд {battery}%) — відео не зупиняється")

        def done():
            self._state = HOVER      # ПІСЛЯ злету просто висимо, не женемось
            self._settle = self._settle_after_takeoff
            self._ever_acquired = False
            print("[flight] у повітрі. HOVER. Натисни G, щоб увімкнути стеження.")
            if self._arm_requested:          # G тиснули ще під час зльоту
                self._arm_requested = False
                self.arm()

        self._run_async("зліт", self._drone.takeoff, done)

    def land(self):
        if self._state == GROUNDED or self._busy:
            return
        print("[flight] саджу")
        self._settle = 0.0
        self._ever_acquired = False
        self._arm_requested = False
        self._state = GROUNDED       # одразу, щоб цикл більше не слав команд
        self._run_async("посадка", self._drone.land, lambda: None)

    def emergency(self):
        # Аварія — єдина команда, яку шлемо СИНХРОННО й без перевірок стану:
        # тут кожна мілісекунда важить більше за плавність картинки.
        print("[flight] АВАРІЯ — мотори вимкнено")
        self._state = GROUNDED
        self._busy = False
        try:
            self._drone.emergency()
        except Exception as e:
            print(f"[flight] аварійна команда не пройшла: {e}")

    def arm(self):
        """Увімкнути автономне стеження.

        ЧОМУ ТУТ ВІДКЛАДЕНИЙ ЗАПИТ: зліт виконується у фоні й може відповідати
        кілька секунд. Якщо натиснути G у цей момент, стан ще GROUNDED — і
        раніше натискання просто ПРОПАДАЛО, доводилось тиснути ще раз. Тепер
        запам'ятовуємо намір і вмикаємо стеження, щойно дрон опиниться в HOVER."""
        if self._state == HOVER:
            self._state = AUTO
            self._time_without_target = 0.0
            self._arm_requested = False
            print("[flight] AUTO — стеження увімкнено")
        elif self._busy or self._state == GROUNDED:
            self._arm_requested = True
            print("[flight] зліт ще триває — стеження увімкнеться автоматично")

    def disarm(self):
        """Вимкнути стеження, лишитись у повітрі."""
        if self._state == AUTO:
            self._state = HOVER
            self._drone.send(Command(0, 0, 0))
            print("[flight] HOVER — стеження вимкнено")

    def toggle(self):
        # Під час зльоту (BUSY) або на землі — трактуємо G як ЗАПИТ увімкнути,
        # а не як вимкнення. Інакше натискання під час зльоту йшло б у
        # disarm() і мовчки пропадало.
        if self._state == AUTO:
            self.disarm()
        else:
            self.arm()

    # ── Основний крок, викликається щокадру ────────────────────────────
    def update(self, command, has_target, dt):
        """command — те, що порахував PID (може бути None);
        has_target — чи є зараз надійна ціль (або її прогноз);
        Повертає команду, яку РЕАЛЬНО відправлено (для показу на екрані)."""
        # Поки виконується зліт/посадка — НІЧОГО не шлемо: RC-команди билися б
        # із самою командою зльоту, яку дрон зараз відпрацьовує.
        if self._busy or self._state == GROUNDED:
            return None

        # Батарея — перевіряємо раз на ~2 с (запит до дрона не безкоштовний).
        self._battery_checked_at += dt
        if self._battery_checked_at >= 2.0:
            self._battery_checked_at = 0.0
            self._last_battery = self._drone.battery()
            if self._last_battery < self._min_battery:
                print(f"[flight] заряд < {self._min_battery}% — примусова посадка")
                self.land()
                return None

        # У режимі висіння PID ігноруємо повністю.
        if self._state == HOVER:
            self._prev_cmd = Command(0.0, 0.0, 0.0)
            self._drone.send(Command(0, 0, 0))
            return Command(0, 0, 0)

        # --- Далі тільки AUTO ---

        # ВІДСТОЮВАННЯ після зльоту: кілька секунд просто висимо. Одразу після
        # зльоту Tello ще стабілізує позицію своєю нижньою камерою; якщо в цей
        # момент почати слати команди, вони б'ються з його вирівнюванням і дрон
        # "пливе" вбік. Даємо йому спокій, тоді починаємо стеження.
        if self._settle > 0.0:
            self._settle -= dt
            self._prev_cmd = Command(0.0, 0.0, 0.0)
            self._drone.send(Command(0, 0, 0))
            return Command(0, 0, 0)

        if has_target and command is not None:
            self._time_without_target = 0.0
            self._ever_acquired = True         # ціль хоч раз захоплена
            safe = self._ramp(self._limit(command), dt)
            self._drone.send(safe)
            return safe

        # Цілі немає: рахуємо, скільки вже.
        self._time_without_target += dt

        # Авто-посадка при втраті — ЛИШЕ якщо ціль КОЛИСЬ була захоплена. Поки
        # ми ще жодного разу не знайшли ціль (щойно зліт, ще не клікнули), НЕ
        # саджаємо — просто висимо й чекаємо. Раніше через це дрон сам сідав за
        # 8 с, якщо ціль не встигли обрати, і "зліт → посадка → зліт" по колу.
        if self._ever_acquired and self._time_without_target >= self._land_after_lost:
            print(f"[flight] цілі немає {self._land_after_lost:.0f} с — саджу")
            self.land()
            return None

        if self._time_without_target >= self._hover_after_lost:
            # ВИСИМО. Ключове рішення безпеки: не продовжуємо рух останньою
            # командою — дрон, що летить наосліп, врізається в стіну.
            self._prev_cmd = Command(0.0, 0.0, 0.0)
            self._drone.send(Command(0, 0, 0))
            return Command(0, 0, 0)

        # Дуже коротка пауза (< 0.5 с) — дотримуємось прогнозу Калмана.
        if command is not None:
            safe = self._ramp(self._limit(command), dt)
            self._drone.send(safe)
            return safe
        self._drone.send(Command(0, 0, 0))
        return Command(0, 0, 0)

    def _ramp(self, target: Command, dt) -> Command:
        """Обмежити ШВИДКІСТЬ НАРОСТАННЯ команди (не саму команду).

        Асиметрично й навмисно:
          • РОЗГІН обмежений — звідси плавність, немає ривків;
          • ГАЛЬМУВАННЯ (у бік нуля) миттєве — якщо треба зупинитись або
            змінити напрямок, дрон робить це без затримки. Плавність не має
            коштувати безпеки."""
        step = self._slew * max(dt, 1e-3)

        def one(new, old):
            if abs(new) <= abs(old) or new * old < 0:
                return new                      # гальмуємо / міняємо напрямок — миттєво
            return max(old - step, min(new, old + step))

        out = Command(yaw=one(target.yaw, self._prev_cmd.yaw),
                      vertical=one(target.vertical, self._prev_cmd.vertical),
                      forward=one(target.forward, self._prev_cmd.forward))
        self._prev_cmd = out
        return out

    # Приведення команд PID до безпечного, але ДІЄВОГО діапазону.
    def _limit(self, c: Command) -> Command:
        vertical = self._shape(c.vertical, self._max_speed)
        vertical = self._apply_altitude(vertical)     # межі стелі/підлоги
        return Command(
            yaw=self._shape(c.yaw, self._max_speed),
            vertical=vertical,
            # Рух уперед/назад — найризикованіший (дрон летить НА тебе). Тому
            # окрема, нижча межа і вимкнення за замовчуванням.
            # Стеля залежить від НАПРЯМКУ: назад — удвічі повільніше.
            forward=(self._shape(c.forward,
                                 self._max_forward if c.forward >= 0 else self._max_backward)
                     if self._enable_forward else 0.0),
        )

    def _apply_altitude(self, vertical):
        """М'які межі висоти (Soft Ceiling/Floor). Біля стелі плавно гасимо
        набір висоти, біля підлоги — зниження; на самій межі блокуємо повністю.
        Так дрон не в'їде в стелю/підлогу при тесті в кімнаті."""
        if not self._altitude_limits:          # вимкнено — вертикаль проходить як є
            return vertical
        if vertical == 0:
            return 0.0
        h = self._drone.height()               # см
        margin = self._height_margin
        if vertical > 0:                        # рух ВГОРУ
            if h >= self._max_height:
                return 0.0                      # уперлись у стелю
            if h > self._max_height - margin:   # у зоні гальмування — плавно гасимо
                vertical *= (self._max_height - h) / margin
        else:                                   # рух ВНИЗ
            if h <= self._min_height:
                return 0.0                      # уперлись у підлогу
            if h < self._min_height + margin:
                vertical *= (h - self._min_height) / margin
        return vertical

    def _shape(self, v, ceiling):
        """ОБРІЗАЄМО зверху, ПІДТЯГУЄМО знизу.

        Раніше тут було МАСШТАБУВАННЯ (v * max_speed/100) поверх уже обмеженого
        PID — подвійне послаблення. Через нього помилка 60px давала в мотори 9,
        а Tello ігнорує все, що менше ~10-15. Звідси й була млявість.

        Тепер: межа — це просто обрізання, а не множник. Плюс компенсація
        "мертвої зони": ненульову, але заслабку команду підтягуємо до min_speed,
        інакше дрон її просто не відпрацює."""
        if v == 0:
            return 0.0
        v = max(-ceiling, min(v, ceiling))        # обрізання зверху
        if abs(v) < self._min_speed:              # підтягування знизу
            v = self._min_speed if v > 0 else -self._min_speed
        return v
