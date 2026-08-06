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
from avis.control.flight import AUTO, GROUNDED, FlightSupervisor
from avis.perception import Detector, FrameSource, KalmanFilter, TargetSelector
from avis.view import Renderer
from avis.view.keys import TerminalKeys

# Клавіші виходу. Esc (27) — надійний вихід у будь-якій розкладці й з CapsLock.
# Літери — best-effort: q/Q (англ./нім.), й/Й (укр./рос.), обидва регістри — CapsLock.
_QUIT_KEYS = {27, ord('q'), ord('Q'), ord('й'), ord('Й')}


class App:
    def __init__(
        self,
        source: FrameSource,            # абстракція (Webcam / Tello / Replay)
        detector: Detector = None,      # None у реплеї (детекції беремо з запису)
        selector: TargetSelector = None,
        tracker: KalmanFilter = None,
        error_calculator: ErrorCalculator = None,
        controller: FollowController = None,
        renderer: Renderer = None,
        supervisor: FlightSupervisor = None,  # None = політ вимкнено (лише перегляд)
        recorder=None,                  # FlightRecorder або None (запис вимкнено)
    ):
        self._supervisor = supervisor
        self._recorder = recorder
        self._keys = TerminalKeys()     # обхід проблеми фокуса вікна на macOS
        self._source = source
        self._detector = detector
        self._selector = selector
        self._tracker = tracker
        self._error_calculator = error_calculator
        self._controller = controller
        self._renderer = renderer

        self._renderer.set_mouse_callback(self._on_mouse)

        # Чи джерело вміє віддавати записані детекції/dt (тобто це реплей).
        self._replaying = hasattr(source, "replayed_detections")
        self._replay_seeded = False  # чи вже відтворили вибір цілі при реплеї

        self._last_detections = []   # останні рамки (для обробника кліку)
        self._last_area = 0.0        # остання відома площа цілі (для дистанції)
        self._prev_time = None       # час попереднього кадру (для dt)
        self._pending_click = None   # відкладений клік (x, y) для надійного вибору
        self._pending_click_ttl = 0  # скільки кадрів ще пробувати обрати ціль
        self._active_target_id = None  # id цілі минулого кадру (ловимо підміну)
        # Підсумкова статистика прогону (для чесного порівняння тюнінгів).
        self._stats = {
            "frames": 0, "VIS": 0, "PRED": 0, "LOST": 0,
            "sum_ex": 0.0, "sum_ey": 0.0,
            "sat_fwd": 0, "sat_yaw": 0, "sat_vert": 0,  # кадрів у сатурації (впертись у стелю)
            "cmd_frames": 0,                             # кадрів, де взагалі були команди
            "jitter": 0.0, "jitter_n": 0,                # дрож: сума |Δcmd| між кадрами
        }
        self._prev_sent = None    # попередня відправлена команда (для дрожу)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # НЕ вибираємо одразу: ставимо "відкладений клік" на кілька кадрів.
            # Мишача подія приходить асинхронно, і рамка могла саме цієї миті
            # зникнути/зсунутись (лаг, мерехтіння). Тому пробуємо кілька кадрів.
            self._pending_click = (x, y)
            self._pending_click_ttl = 10

    def run(self):
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break

                # --- Крок 0: dt (скільки секунд минуло з минулого кадру) ---
                if self._replaying:
                    # РЕПЛЕЙ: беремо dt із запису, а не реальний час (інакше на
                    # швидкому прогоні dt був би крихітний і зламав би Kalman).
                    dt = self._source.replayed_dt()
                else:
                    now = time.time()
                    dt = 0.0 if self._prev_time is None else now - self._prev_time
                    self._prev_time = now
                # Затиск dt. Верхню межу підняли 0.1 → 0.3, бо YOLO на CPU дає
                # 4-8 FPS, і при старому затиску ВСІ таймери йшли вдвічі
                # повільніше за реальний час: пільга втрати 3 с ставала 6 с,
                # відстоювання 5 с — 10 с, а дрон летів наосліп удвічі довше.
                dt = max(1e-3, min(dt, 0.3))

                # --- Крок 1: детекція (або записані детекції в реплеї) ---
                t0 = time.time()
                if self._replaying:
                    detections = self._source.replayed_detections()  # YOLO не запускаємо
                else:
                    detections = self._detector.detect(frame)
                self._last_detections = detections
                proc = time.time() - t0
                fps = 1 / proc if proc > 1e-6 else 999.0

                h, w = frame.shape[:2]

                # РЕПЛЕЙ: відтворюємо вибір цілі (кліку мишею немає). Один раз —
                # на першому кадрі, де в записі вже була ціль. Далі селектор
                # працює сам, і якщо з новим тюнінгом він її загубить — це
                # реальна поведінка, яку ми й хочемо побачити.
                if self._replaying and not self._replay_seeded:
                    tid = self._source.replayed_target_id()
                    if tid is not None:
                        seed = next((d for d in detections if d.id == tid), None)
                        if seed is not None:
                            self._selector.select_target(seed)
                            self._replay_seeded = True

                # ВІДКЛАДЕНИЙ КЛІК: пробуємо обрати ціль ще кілька кадрів після
                # натискання. Щойно вдалося (або вичерпали спроби) — забуваємо.
                if self._pending_click is not None:
                    picked = self._selector.select_at(*self._pending_click, detections)
                    self._pending_click_ttl -= 1
                    done = picked is not None or self._pending_click_ttl <= 0
                    if picked is not None:
                        # Діагностика: одразу видно, що ціль справді обрано
                        # і що далі треба натиснути G, щоб полетіти за нею.
                        print(f"[app] ціль обрано: #{picked.id} ({picked.name}) — тисни G")
                    elif done:
                        print("[app] клік повз — жодної рамки поруч")
                    if done:
                        self._pending_click = None

                # --- Крок 2: де ціль + згладжування/пам'ять фільтром ---
                # Спершу ПРОГНОЗ Калмана (пріор цього кадру). Він потрібен двічі:
                #   а) як позиція для повторного захоплення цілі в resolve();
                #   б) як позиція для керування, коли ціль перекрита/зникла.
                predicted = self._tracker.predict(dt) if self._tracker.initialized else None

                # ЗАТИСК ПРОГНОЗУ В МЕЖІ КАДРУ. Якщо ціль вийшла ЗА кадр (а не
                # просто сховалась), прогноз екстраполює назовні: x=700, 900...
                # Тоді PID ганявся б за фантомом поза екраном, а повторне
                # захоплення шукало б ціль там, де рамки не бувають — і не
                # знайшло б її при поверненні. Затиснувши прогноз до краю, ми:
                #   • змушуємо дрон довертатись у бік, де ціль зникла;
                #   • шукаємо повернення саме біля того краю (там, звідки вона
                #     й повернеться). Це і є виправлення "зникла з кадру".
                off_frame = False
                if predicted is not None:
                    px, py = predicted
                    cpx = min(max(px, 0.0), w - 1.0)
                    cpy = min(max(py, 0.0), h - 1.0)
                    off_frame = (cpx != px) or (cpy != py)   # прогноз був за межами
                    predicted = (cpx, cpy)

                # resolve сам вирішує: надійно видно / оклюзія-втрата / зникла.
                target, lost = self._selector.resolve(detections, predicted, dt)

                center = None
                if target is not None:
                    # ЗМІНА ЦІЛІ (перезахоплення на інший об'єкт) — скидаємо
                    # фільтр і регулятор. Інакше позиція стрибає на сотні
                    # пікселів, D-складова PID дає величезний спайк, а Калман
                    # трактує стрибок як шалену швидкість — звідси різкі ривки.
                    if self._active_target_id is not None and target.id != self._active_target_id:
                        print(f"[app] ціль змінилась #{self._active_target_id} → #{target.id}"
                              f" ({target.name}) — скидаю фільтр і регулятор")
                        self._tracker.reset()
                        self._controller.reset()
                    self._active_target_id = target.id

                    # Ціль надійно видно (або перезахоплено) — коригуємо фільтр виміром.
                    # `*target.center` розкладає кортеж (cx, cy) на два аргументи.
                    center = self._tracker.correct(*target.center)
                    self._last_area = target.area
                    status = "VIS"
                elif lost:
                    # Оклюзія (<25%) або коротке зникнення: рамці не віримо —
                    # летимо за ПРОГНОЗОМ Калмана (predicted може бути None на
                    # найпершому кадрі до ініціалізації).
                    center = predicted
                    # "OFF" = ціль пішла ЗА межі кадру (прогноз затиснуто до краю),
                    # "PRED" = ціль десь у кадрі, але перекрита.
                    status = "PRED"
                    if off_frame:
                        print("  ! ціль за межами кадру — тримаємо напрямок на край")
                else:
                    # Ціль зникла надовго: скидаємо фільтр і регулятор.
                    self._tracker.reset()
                    self._controller.reset()
                    self._active_target_id = None
                    status = "LOST"

                # --- Крок 3: помилка → регулятор → команди ---
                cmd = None
                if center is not None:
                    cx, cy = center
                    error = self._error_calculator.compute(cx, cy, self._last_area, w, h)
                    # ДОВІРА ДО ДИСТАНЦІЇ лише у VIS. У PRED площа рамки —
                    # ЗАСТАРІЛА (остання побачена), бо свіжого виміру немає. Раніше
                    # через це дрон у PRED вважав "ціль далеко" і летів уперед на
                    # повній — саме так він і в'їжджав у людину. Тепер у PRED
                    # керуємо лише напрямком, а дистанцію не чіпаємо.
                    cmd = self._controller.update(error, dt, trust_distance=(status == "VIS"))

                # --- Крок 3b: safety-фільтр і ВІДПРАВКА в мотори ---
                # Сюди PID віддає "куди хочу летіти", а supervisor вирішує, чи
                # МОЖНА (стан польоту, втрата цілі, заряд) і приглушує швидкості.
                sent = None
                if self._supervisor is not None:
                    # ЧЕСНИЙ has_target: True ЛИШЕ коли ціль реально видно (VIS).
                    # Раніше сюди йшло `center is not None`, а в PRED центр — це
                    # прогноз Калмана, тобто НЕ None. Через це supervisor вважав,
                    # що ціль є, і його захист (вис через 0.5 с, посадка через 8 с)
                    # НЕ СПРАЦЬОВУВАВ ЖОДНОГО РАЗУ — дрон крутився за привидом
                    # на повній швидкості весь пільговий період.
                    sent = self._supervisor.update(cmd, target is not None, dt)

                if cmd is not None:
                    line = (f"[{status:4}] ex={error.error_x:+7.0f} ey={error.error_y:+7.0f}"
                            f" area={error.area:8.0f}   |   yaw={cmd.yaw:+6.1f}"
                            f" vert={cmd.vertical:+6.1f} fwd={cmd.forward:+6.1f}")
                    if sent is not None:
                        line += (f"   → МОТОРИ yaw={sent.yaw:+5.1f}"
                                 f" vert={sent.vertical:+5.1f} fwd={sent.forward:+5.1f}")
                    print(line)

                # --- Крок 3c: статистика + ЗАПИС (для аналізу й реплею) ---
                self._accumulate_stats(status, center, w, h, cmd, sent)
                if self._recorder is not None:
                    # ЧИСТИЙ кадр (до малювання рамок) + вхід пайплайну + базова
                    # лінія (що реально пішло в мотори, стан, заряд).
                    sup = self._supervisor
                    self._recorder.write(
                        frame, dt, detections, self._selector.target_id,
                        sent=sent,
                        state=None if sup is None else sup.state,
                        battery=None if sup is None else sup.battery,
                    )

                # --- Крок 4: візуалізація ---
                # Передаємо статус і гіпотезу Калмана, щоб при оклюзії на екрані
                # було видно, ДЕ система вважає ціль (а не порожнеча, як раніше).
                # До статусу трекінгу додаємо СТАН ПОЛЬОТУ (HOVER/AUTO) — щоб
                # одразу бачити, стежить дрон чи лише висить, не лізучи в консоль.
                shown = status
                if self._supervisor is not None:
                    shown = f"{status}|{self._supervisor.state}"
                self._renderer.draw(frame, detections, self._selector.target_id,
                                    fps, shown, predicted)

                # --- Крок 5: клавіатура / закриття вікна ---
                key = cv2.waitKey(1)
                typed = self._keys.get()          # клавіші й з терміналу теж
                if typed:
                    key = 27 if typed in ('\x1b', '\x03') else ord(typed.lower())

                if key != -1 and self._handle_key(key):
                    break
                if not self._renderer.is_open():
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C")
        finally:
            # Порядок важливий: спершу посадити дрон, потім відпускати ресурси.
            if self._supervisor is not None and self._supervisor.state != GROUNDED:
                print("Завершення — саджу дрон ...")
                self._supervisor.land()
            if self._recorder is not None:
                self._recorder.close()
            self._keys.restore()
            self._source.release()
            cv2.destroyAllWindows()
            self._print_summary()

    def _accumulate_stats(self, status, center, w, h, cmd=None, sent=None):
        """Копить підсумок прогону — метрики для ЧЕСНОГО порівняння тюнінгів:
        стани, помилка по осях, САТУРАЦІЯ (чи впирається в стелю) і ДРОЖ команд."""
        s = self._stats
        s["frames"] += 1
        s[status] = s.get(status, 0) + 1
        if center is not None:
            s["sum_ex"] += abs(center[0] - w / 2)
            s["sum_ey"] += abs(center[1] - h / 2)

        if cmd is not None and sent is not None:
            s["cmd_frames"] += 1
            # САТУРАЦІЯ: supervisor обрізав команду (|відправлено| < |просив|).
            # Не треба знати стелі — сам факт обрізання і є сатурація.
            eps = 0.5
            if abs(sent.forward) + eps < abs(cmd.forward):
                s["sat_fwd"] += 1
            if abs(sent.yaw) + eps < abs(cmd.yaw):
                s["sat_yaw"] += 1
            if abs(sent.vertical) + eps < abs(cmd.vertical):
                s["sat_vert"] += 1
            # ДРОЖ: наскільки команда скакнула з минулого кадру. Велика — дрон
            # смикається (зазвичай через deadzone/min_speed або завеликий kp).
            if self._prev_sent is not None:
                s["jitter"] += (abs(sent.yaw - self._prev_sent.yaw)
                                + abs(sent.vertical - self._prev_sent.vertical)
                                + abs(sent.forward - self._prev_sent.forward))
                s["jitter_n"] += 1
            self._prev_sent = sent

    def _print_summary(self):
        """Друкує підсумок наприкінці прогону. Саме ці числа порівнюй між
        прогонами реплею, щоб оцінити тюнінг об'єктивно, а не "на око"."""
        s = self._stats
        n = s["frames"]
        if n == 0:
            return
        tracked = s["VIS"] + s["PRED"]           # кадри, де ціль вело
        avg_ex = s["sum_ex"] / tracked if tracked else 0.0
        avg_ey = s["sum_ey"] / tracked if tracked else 0.0
        cf = s["cmd_frames"]
        print("\n" + "─" * 60)
        print(f"ПІДСУМОК: {n} кадрів   VIS={s['VIS']} PRED={s['PRED']} LOST={s['LOST']}")
        print(f"  утримання цілі : {100 * tracked / n:.0f}%")
        print(f"  сер. помилка   : ex={avg_ex:.0f}px  ey={avg_ey:.0f}px")
        if cf:
            print(f"  сатурація      : вперед {100 * s['sat_fwd'] / cf:.0f}%  "
                  f"поворот {100 * s['sat_yaw'] / cf:.0f}%  "
                  f"вертикаль {100 * s['sat_vert'] / cf:.0f}%")
            if s["jitter_n"]:
                print(f"  дрож команд    : {s['jitter'] / s['jitter_n']:.1f} од/кадр")
        print("  ── менша помилка + менша дрож = кращий тюнінг")
        print("  ── висока сатурація вперед = стеля max_forward тримає, підіймай її")
        print("─" * 60)

    def _handle_key(self, key):
        """Обробити клавішу. Повертає True, якщо треба вийти з циклу.

        ВАЖЛИВО про код клавіші: cv2.waitKey на macOS/Qt повертає значення із
        зайвими СТАРШИМИ БІТАМИ (напр. 1048679 замість 103 для 'g'). Тому
        звіряємо ОБИДВА варіанти — сирий код і обрізаний маскою 0xFF. Без цього
        клавіші не спрацьовували, щойно фокус переходив на вікно OpenCV (тобто
        одразу після кліку по цілі) — і дрон не вмикав стеження."""
        codes = {key, key & 0xFF}

        if codes & _QUIT_KEYS:
            return True

        if self._supervisor is None:
            return False                          # режим перегляду — політ вимкнено

        if ord('t') in codes:
            self._supervisor.takeoff()
        elif ord('l') in codes:
            self._supervisor.land()
        elif ord('g') in codes:
            self._supervisor.toggle()             # увімкнути/вимкнути стеження
        elif ord('x') in codes:
            self._supervisor.emergency()
        else:
            # Клавіша не розпізнана — друкуємо код, щоб було видно, що натиск
            # ДІЙШОВ до програми (а не загубився через фокус вікна).
            print(f"[app] невідома клавіша (код {key}) — T зліт, G стеження, L посадка")
        return False
