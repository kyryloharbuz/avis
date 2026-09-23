# App — ОРКЕСТРАТОР. Робить рівно три речі:
#   1. крутить цикл кадрів і вирішує, коли зупинитись;
#   2. з'єднує ввід/вивід (камера, вікно, клавіші, запис) із конвеєром;
#   3. передає команду конвеєра в safety-фільтр і далі в мотори.
#
# Уся ОБЧИСЛЮВАЛЬНА логіка живе окремо:
#   TrackingPipeline — детекція, трекінг, керування (avis/pipeline.py)
#   FlightSupervisor — безпека польоту (avis/control/flight.py)
#   RunStats, FollowScore — метрики (avis/stats.py, avis/metrics.py)
#
# Раніше все це було всередині run() — метод на 150 рядків, який доводилось
# читати цілком, щоб зрозуміти будь-яку частину. Тепер App не знає, ЯК
# влаштований трекінг, а конвеєр не знає про вікна й клавіші.

import time

import cv2  # потрібен лише для констант вводу вікна (клавіші, миша)

from avis.control.flight import GROUNDED, FlightSupervisor
from avis.metrics import FollowScore
from avis.perception import FrameSource
from avis.pipeline import VIS, TrackingPipeline
from avis.stats import RunStats
from avis.view import Renderer
from avis.view.keys import TerminalKeys

# Клавіші виходу. Esc (27) — надійний у будь-якій розкладці й з CapsLock.
# Літери best-effort: q/Q (англ./нім.), й/Й (укр./рос.), обидва регістри.
_QUIT_KEYS = {27, ord('q'), ord('Q'), ord('й'), ord('Й')}

# Затиск dt. Верхня межа 0.3 с (а не 0.1): YOLO на CPU дає 4-8 FPS, і при
# тіснішому затиску ВСІ таймери йшли повільніше за реальний час — пільга
# втрати цілі 3 с ставала 6 с, а дрон летів наосліп удвічі довше.
_DT_MIN, _DT_MAX = 1e-3, 0.3


class App:
    def __init__(self, source: FrameSource, detector=None, selector=None,
                 tracker=None, error_calculator=None, controller=None,
                 renderer: Renderer = None,
                 supervisor: FlightSupervisor = None, recorder=None):
        self._source = source
        self._renderer = renderer
        self._supervisor = supervisor
        self._recorder = recorder
        self._keys = TerminalKeys()   # обхід проблеми фокуса вікна на macOS

        self._pipeline = TrackingPipeline(
            detector, selector, tracker, error_calculator, controller,
            on_event=lambda msg: print(f"[app] {msg}"))

        self._renderer.set_mouse_callback(self._on_mouse)

        # Реплей упізнаємо за наявністю додаткових методів у джерела.
        self._replaying = hasattr(source, "replayed_detections")

        self._prev_time = None
        self._pending_click = None    # відкладений клік (x, y)
        self._pending_click_ttl = 0
        self._last_key = None         # дебаунс клавіш
        self._last_key_time = 0.0

        self._stats = RunStats()
        # Уставку метрики беремо З КОНТРОЛЕРА — одне джерело істини, щоб
        # метрика міряла рівно те, чим керує система.
        self._follow = FollowScore(
            target_m=getattr(controller, "distance_target_m", 1.6),
            tolerance_m=3 * getattr(controller, "distance_deadzone_m", 0.25))

    # ── Ввід ───────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # НЕ вибираємо одразу: рамка могла саме цієї миті зникнути або
            # зсунутись (лаг, мерехтіння). Пробуємо кілька кадрів поспіль.
            self._pending_click = (x, y)
            self._pending_click_ttl = 10

    # ── Головний цикл ──────────────────────────────────────────────────
    def run(self):
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break

                dt = self._frame_dt()
                detections = (self._source.replayed_detections()
                              if self._replaying else None)

                result = self._pipeline.process(frame, dt, detections)

                self._seed_replay_target(result)
                self._resolve_pending_click(result)

                # Safety-фільтр вирішує, чи МОЖНА виконати те, чого хоче
                # регулятор, і приглушує швидкості перед моторами.
                sent = None
                if self._supervisor is not None:
                    sent = self._supervisor.update(result.command,
                                                   result.has_target, dt,
                                                   blind=result.blind)

                h, w = frame.shape[:2]
                self._report(result, sent)
                self._collect(result, sent, w, h)
                self._record(frame, dt, result, sent)
                self._draw(frame, result)

                if self._handle_input():
                    break
                if not self._renderer.is_open():
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C")
        finally:
            self._shutdown()

    # ── Кроки циклу (кожен — одна думка) ───────────────────────────────
    def _frame_dt(self):
        """У реплеї беремо dt із ЗАПИСУ: на швидкому прогоні реальний час між
        кадрами крихітний, і Kalman/PID працювали б у хибному масштабі часу."""
        if self._replaying:
            dt = self._source.replayed_dt()
        else:
            now = time.time()
            dt = 0.0 if self._prev_time is None else now - self._prev_time
            self._prev_time = now
        return max(_DT_MIN, min(dt, _DT_MAX))

    def _seed_replay_target(self, result):
        """У реплеї кліку мишею немає — відтворюємо вибір оператора за
        записаним target_id.

        ЧОМУ НЕ ОДИН РАЗ. Спершу тут стояв прапорець "засіяно" на весь прогін,
        і це ламало весь стенд: щойно селектор скидав ціль (а він скидає після
        межі очікування), відновити її не було кому — реплей до кінця йшов у
        LOST. Замір: утримання цілі 12% при 77% у тому самому живому польоті,
        тобто стенд показував не той політ, який записали, і тюнити на ньому
        було безглуздо.

        Правильна умова — не "чи вже сіяли", а "чи є ціль ЗАРАЗ": у записі
        видно, що оператор у цей момент ціль вів, тож відтворюємо це щоразу,
        коли конвеєр лишився без цілі."""
        if not self._replaying or self._pipeline.target_id is not None:
            return
        tid = self._source.replayed_target_id()
        if tid is None:
            return
        seed = next((d for d in result.detections if d.id == tid), None)
        if seed is not None:
            self._pipeline.select_target(seed)

    def _resolve_pending_click(self, result):
        if self._pending_click is None:
            return
        picked = self._pipeline.select_at(*self._pending_click, result.detections)
        self._pending_click_ttl -= 1
        if picked is not None:
            print(f"[app] ціль обрано: #{picked.id} ({picked.name}) — тисни G")
        elif self._pending_click_ttl <= 0:
            print("[app] клік повз — жодної рамки поруч")
        if picked is not None or self._pending_click_ttl <= 0:
            self._pending_click = None

    def _report(self, result, sent):
        if result.command is None:
            return
        e, c = result.error, result.command
        dist = ("  ---" if e.distance_m is None
                else f"{e.distance_m:4.2f}м" + ("!" if e.box_clipped else " "))
        line = (f"[{result.status:4}] ex={e.error_x:+7.0f} ey={e.error_y:+7.0f}"
                f" d={dist}   |   yaw={c.yaw:+6.1f}"
                f" vert={c.vertical:+6.1f} fwd={c.forward:+6.1f}")
        if sent is not None:
            line += (f"   → МОТОРИ yaw={sent.yaw:+5.1f}"
                     f" vert={sent.vertical:+5.1f} fwd={sent.forward:+5.1f}")
        print(line)

    def _collect(self, result, sent, w, h):
        self._stats.add(result, sent, w, h)
        # Follow-скор рахуємо ЛИШЕ по реально видимій цілі: у PRED позиція —
        # гіпотеза Калмана, зараховувати її як успіх було б самообманом.
        #
        # Дистанцію віддаємо ГОТОВУ — ту саму, за якою працює регулятор. Раніше
        # метрика рахувала свою, з висоти рамки, і міряла не те, чим керує
        # система (див. коментар у FollowScore.add).
        visible = result.status == VIS
        self._follow.add(result.center if visible else None,
                         result.error.distance_m if result.error else None,
                         w, h)

    def _record(self, frame, dt, result, sent):
        if self._recorder is None:
            return
        sup = self._supervisor
        # ЧИСТИЙ кадр (до малювання) + вхід конвеєра + базова лінія команд.
        self._recorder.write(frame, dt, result.detections, self._pipeline.target_id,
                             sent=sent,
                             state=None if sup is None else sup.state,
                             battery=None if sup is None else sup.battery)

    def _draw(self, frame, result):
        fps = (1 / result.detect_seconds) if result.detect_seconds > 1e-6 else 999.0
        # До статусу трекінгу додаємо стан польоту — одразу видно, стежить
        # дрон чи лише висить, без заглядання в консоль.
        shown = result.status
        if self._supervisor is not None:
            shown = f"{result.status}|{self._supervisor.state}"
        # Побитий кадр показуємо ЯВНО. Інакше на екрані видно лише наслідок
        # ("ціль зникла"), і причину — сміття у відео — легко списати на YOLO.
        if result.frame_corrupt:
            shown += "|NOISE"
        self._renderer.draw(frame, result.detections, self._pipeline.target_id,
                            fps, shown, result.predicted)

    def _handle_input(self) -> bool:
        key = cv2.waitKey(1)
        typed = self._keys.get()      # клавіші й з терміналу теж (macOS)
        if typed:
            key = 27 if typed in ('\x1b', '\x03') else ord(typed.lower())
        return key != -1 and self._handle_key(key)

    def _handle_key(self, key) -> bool:
        """cv2.waitKey на macOS/Qt повертає код із зайвими СТАРШИМИ БІТАМИ
        (1048679 замість 103 для 'g'), тож звіряємо обидві форми. Без цього
        клавіші не працювали, щойно фокус переходив на вікно OpenCV."""
        codes = {key, key & 0xFF}

        # Дебаунс: одне натискання може прийти двічі (вікно + термінал). Для G
        # це означало "увімкнув і одразу вимкнув" — здавалось, що не працює.
        now = time.time()
        norm = key & 0xFF
        if norm == self._last_key and (now - self._last_key_time) < 0.4:
            return False
        self._last_key, self._last_key_time = norm, now

        if codes & _QUIT_KEYS:
            return True
        if self._supervisor is None:
            return False              # режим перегляду — політ вимкнено

        if ord('t') in codes:
            self._supervisor.takeoff()
        elif ord('l') in codes:
            self._supervisor.land()
        elif ord('g') in codes:
            self._supervisor.toggle()
        elif ord('x') in codes:
            self._supervisor.emergency()
        else:
            print(f"[app] невідома клавіша (код {key}) — T зліт, G стеження, L посадка")
        return False

    def _shutdown(self):
        """Порядок важливий: спершу посадити дрон, потім відпускати ресурси."""
        if self._supervisor is not None and self._supervisor.state != GROUNDED:
            print("Завершення — саджу дрон ...")
            self._supervisor.land()
        if self._recorder is not None:
            self._recorder.close()
        self._keys.restore()
        self._source.release()
        cv2.destroyAllWindows()

        lines = self._stats.report()
        if lines:
            print("\n" + "─" * 60)
            for line in lines:
                print(line)
            for line in self._follow.report():
                print("  " + line)
            print("─" * 60)
