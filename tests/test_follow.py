# Тести пакета avis.control.follow. Пишуться ПАРАЛЕЛЬНО з кожною фазою.
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PASS = 0
_FAIL = 0
_FAILURES = []


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {name}")
    else:
        _FAIL += 1
        _FAILURES.append(name)
        print(f"  ✗ FAIL: {name}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ── ФАЗА 2: конфіг ───────────────────────────────────────────────────────
def test_config():
    print("\n[ FollowConfig — валідація ловить реальні баги ]")
    from avis.control.follow.config import FollowConfig

    c = FollowConfig()
    check("дефолти валідні", c.distance_target_m == 0.75)

    def raises(**kw):
        try:
            FollowConfig(**kw)
            return False
        except ValueError:
            return True

    check("зона спокою ширша за уставку → помилка",
          raises(distance_target_m=0.2, distance_deadzone_m=0.5))
    check("align_full >= align_stop → помилка",
          raises(align_full=0.9, align_stop=0.5))
    check("align_stop > 1 → помилка", raises(align_stop=1.5))
    check("відʼємна уставка → помилка", raises(distance_target_m=-1))
    check("retreat_factor поза (0,1] → помилка", raises(retreat_factor=0))
    check("відʼємна зона спокою → помилка", raises(yaw_deadzone_px=-5))


# ── ФАЗА 3: базова вісь ──────────────────────────────────────────────────
def test_axis():
    print("\n[ Axis — ЄДИНА мʼяка зона спокою ]")
    from avis.control.follow.axis import Axis

    a = Axis(kp=1.0, kd=0.0, deadzone=10.0)
    check("всередині зони → рівно 0", a.shape(5) == 0.0)
    check("на самій межі → 0", a.shape(10) == 0.0)
    check("за межею → починається з нуля, без сходинки", approx(a.shape(11), 1.0))
    check("симетрично для відʼємних", approx(a.shape(-11), -1.0))
    check("далеко → помилка мінус зона", approx(a.shape(100), 90.0))

    # головна властивість: НЕМАЄ стрибка на межі
    just_out = abs(a.update(10.001, 0.1))
    check("на межі команда ~0 (немає ривка)", just_out < 0.01)

    a.reset()
    check("hold() повертає 0", a.hold() == 0.0)


# ── ФАЗА 4: осі ──────────────────────────────────────────────────────────
def _err(ex=0.0, ey=0.0, dist=1.6, clipped=False, w=960, h=720, fill=0.1):
    from avis.models import ControlError
    return ControlError(error_x=ex, error_y=ey, area=0.0,
                        norm_x=ex / (w / 2), norm_y=ey / (h / 2),
                        distance_m=dist, box_clipped=clipped, frame_fill=fill)


def test_yaw_axis():
    print("\n[ YawAxis ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.yaw import YawAxis

    a = YawAxis(FollowConfig())
    check("у центрі → 0", a.command(_err(ex=0), 0.1) == 0.0)
    check("у зоні спокою (20px) → 0", a.command(_err(ex=20), 0.1) == 0.0)
    out = a.command(_err(ex=200), 0.1)
    check("праворуч → поворот праворуч (+)", out > 0)
    a.reset()
    check("ліворуч → поворот ліворуч (-)", a.command(_err(ex=-200), 0.1) < 0)


def test_vertical_axis():
    print("\n[ VerticalAxis — адаптивний рівень і заморозка ]")
    from avis.control.follow.config import FollowConfig as _C
    from avis.control.follow.vertical import VerticalAxis as _V
    # ВИМКНЕНА за замовчуванням — через коливання від зворотного зв'язку
    off = _V(_C())
    check("вимкнена за замовчуванням → завжди 0",
          off.command(_err(ey=-300), 0.15, distance_rate=0.0) == 0.0)
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.vertical import VerticalAxis

    # 1. Постійний зсув НЕ має підіймати дрон
    a = VerticalAxis(FollowConfig(vertical_enabled=True))
    for _ in range(60):
        out = a.command(_err(ey=-130), 0.15, distance_rate=0.0)
    check("постійний зсув -130px → команди немає", abs(out) < 1e-6)

    # 2. Присідання (дистанція стала) — реагує
    for _ in range(4):
        out = a.command(_err(ey=-30), 0.15, distance_rate=0.0)
    check("ціль опустилась → дрон знижується", out < 0)

    # 3. Підліт (дистанція швидко змінюється) — заморожено
    b = VerticalAxis(FollowConfig(vertical_enabled=True))
    for _ in range(40):
        b.command(_err(ey=-130), 0.15, distance_rate=0.0)
    out = b.command(_err(ey=-30), 0.15, distance_rate=-0.8)
    check("зміна дистанції → вертикаль заморожено", out == 0.0)

    # 4. Обрізана рамка — теж заморожено
    c = VerticalAxis(FollowConfig(vertical_enabled=True))
    for _ in range(40):
        c.command(_err(ey=-130), 0.15, distance_rate=0.0)
    check("обрізана рамка → заморожено",
          c.command(_err(ey=-30, clipped=True), 0.15, distance_rate=0.0) == 0.0)


def test_distance_axis():
    print("\n[ DistanceAxis — ОДНА уставка, лише метри ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.distance import DistanceAxis

    cfg = FollowConfig()
    a = DistanceAxis(cfg)
    check("на уставці → 0", a.command(_err(dist=0.75), 0.1) == 0.0)
    a.reset()
    check("у зоні спокою → 0", a.command(_err(dist=0.90), 0.1) == 0.0)
    a.reset()
    check("далеко → вперед", a.command(_err(dist=2.0), 0.1) > 0)
    a.reset()
    check("близько → назад", a.command(_err(dist=0.30), 0.1) < 0)

    # ЛІНІЙНІСТЬ — головна перевага метрів над площею
    x1 = DistanceAxis(cfg).command(_err(dist=0.75 + 0.20 + 1.0), 0.1)
    x2 = DistanceAxis(cfg).command(_err(dist=0.75 + 0.20 + 2.0), 0.1)
    check("удвічі більша помилка → удвічі більша команда", approx(x2 / x1, 2.0, 0.05))

    # Асиметрія й обрізання
    fwd = DistanceAxis(cfg).command(_err(dist=2.0), 0.1)
    back = DistanceAxis(cfg).command(_err(dist=0.2), 0.1)
    check("відступ повільніший за наступ", abs(back) < abs(fwd))
    check("вимір неможливий → вперед заборонено",
          DistanceAxis(cfg).command(_err(dist=2.0, clipped=True), 0.1) <= 0)
    check("вимір неможливий → назад дозволено",
          DistanceAxis(cfg).command(_err(dist=0.2, clipped=True), 0.1) < 0)
    check("немає виміру → 0", DistanceAxis(cfg).command(_err(dist=None), 0.1) == 0.0)


def test_alignment_gate():
    print("\n[ AlignmentGate ]")
    from avis.control.follow.alignment import AlignmentGate
    from avis.control.follow.config import FollowConfig

    g = AlignmentGate(FollowConfig())
    check("по центру → повний хід", g.factor(0.0) == 1.0)
    check("у межах align_full → повний хід", g.factor(0.30) == 1.0)
    check("на краю → хід заборонено", g.factor(0.9) == 0.0)
    mid = g.factor(0.55)
    check("між порогами → частковий хід", 0 < mid < 1)
    check("симетрично", g.factor(-0.55) == mid)


# ── ФАЗА 5: координатор ──────────────────────────────────────────────────
def test_controller():
    print("\n[ FollowController — координація осей ]")
    from avis.control.follow import FollowController

    c = FollowController()
    out = c.update(_err(ex=200, dist=3.0), 0.1)
    check("ціль далеко й збоку → і поворот, і рух уперед", out.yaw > 0 and out.forward > 0)

    c2 = FollowController()
    edge = c2.update(_err(ex=440, dist=3.0), 0.1)
    check("ціль на краю → лише поворот, без руху вперед",
          edge.yaw > 0 and edge.forward == 0.0)

    c3 = FollowController()
    pred = c3.update(_err(ex=200, dist=3.0), 0.1, trust_distance=False)
    check("PRED → дистанцією не керуємо", pred.forward == 0.0)
    check("PRED → напрямок усе одно ведемо", pred.yaw > 0)

    c4 = FollowController(yaw_kp=0.9)
    check("точкове перевизначення параметра працює", c4.config.yaw_kp == 0.9)
    check("уставка доступна назовні (для метрики)", c4.distance_target_m == 0.75)


# ── ФАЗА 7: ЕФЕКТИВНІСТЬ на реальних записах ─────────────────────────────
#
# ЦЕ НАЙВАЖЛИВІШИЙ ТЕСТ У ПРОЄКТІ. Раніше 92 юніт-тести проходили, поки дрон
# фізично не міг зрушити: команда ніколи не досягала величини, яку Tello
# відпрацьовує (~15), і не трималась достатньо довго. Юніт-тести перевіряли
# формули, але НЕ перевіряли, що система дає ДІЄВИЙ результат.
# Тепер перевіряємо саме це — на справжніх польотних даних.

def _run_recording(name):
    """Прогнати запис через повний конвеєр і зібрати команди в мотори."""
    import contextlib as _c
    import io as _io
    from avis.app import App
    from avis.control import ErrorCalculator
    from avis.control.drone import Drone
    from avis.control.flight import FlightSupervisor
    from avis.control.follow import FollowController
    from avis.distance import DistanceEstimator
    from avis.perception import KalmanFilter, TargetSelector
    from avis.replay import ReplaySource
    from avis.view import HeadlessRenderer

    class Spy(Drone):
        def __init__(self): self.sent = []
        def takeoff(self): pass
        def land(self): pass
        def emergency(self): pass
        def send(self, c): self.sent.append(c)
        def battery(self): return 90
        def height(self): return 100

    spy = Spy()
    app = App(source=ReplaySource(name), detector=None, selector=TargetSelector(),
              tracker=KalmanFilter(),
              error_calculator=ErrorCalculator(estimator=DistanceEstimator()),
              controller=FollowController(), renderer=HeadlessRenderer(),
              supervisor=FlightSupervisor(spy, enable_forward=True,
                                          start_armed=True, settle_after_takeoff=0))
    with _c.redirect_stdout(_io.StringIO()):
        app.run()
    return spy.sent


def _longest_run(values, threshold):
    best = cur = 0
    for v in values:
        cur = cur + 1 if abs(v) >= threshold else 0
        best = max(best, cur)
    return best


def test_effectiveness():
    print("\n[ ЕФЕКТИВНІСТЬ на реальних польотах ]")
    import glob
    import os

    # Беремо лише записи, де політ РЕАЛЬНО був. Інакше тест ловить обірвані
    # запуски на 20-30 кадрів (запустив, глянув картинку, закрив) і провалюється
    # не тому, що керування погане, а тому, що дрон тоді й не злітав.
    def is_real_flight(path, min_auto_frames=200):
        try:
            with open(path) as fh:
                return sum(1 for line in fh if '"state": "AUTO"' in line) >= min_auto_frames
        except OSError:
            return False

    recs = [p for p in sorted(glob.glob("recordings/*.jsonl")) if is_real_flight(p)]
    if not recs:
        print("  (справжніх польотів у записах немає — тест пропущено)")
        return

    EFFECTIVE = 15      # нижче цього Tello фізично не рухається
    MIN_HOLD = 5        # команда має триматись хоча б кілька кадрів підряд

    total_fwd_ok = total_yaw_ok = 0
    for rec in recs[-3:]:
        name = os.path.basename(rec)[:-6]
        sent = _run_recording(name)
        if not sent:
            continue
        fwd = [c.forward for c in sent]
        yaw = [c.yaw for c in sent]
        f_run = _longest_run(fwd, EFFECTIVE)
        y_run = _longest_run(yaw, EFFECTIVE)
        f_pct = 100 * sum(1 for v in fwd if abs(v) >= EFFECTIVE) / len(fwd)
        print(f"    {name[7:]}: вперед {f_pct:.0f}% дієвих, "
              f"найдовше {f_run} кадрів; поворот {y_run} кадрів")
        total_fwd_ok += f_run >= MIN_HOLD
        total_yaw_ok += y_run >= MIN_HOLD

    check("вісь ВПЕРЕД дає дієві команди тривало", total_fwd_ok >= 2)
    check("вісь ПОВОРОТУ дає дієві команди тривало", total_yaw_ok >= 2)


def test_no_dual_setpoint():
    """Регресійний запобіжник проти бага, що зламав рух за ціллю:
    у системі має бути ОДНА уставка дистанції, а не дві."""
    print("\n[ Одне джерело істини для дистанції ]")
    from avis.control.follow import FollowController
    from avis.control.follow.config import FollowConfig

    c = FollowController()
    check("уставка лише в конфігу", hasattr(c.config, "distance_target_m"))
    check("немає окремої уставки по площі", not hasattr(c, "_desired_area"))
    check("немає резервної гілки по площі",
          not any("area" in f for f in FollowConfig.__dataclass_fields__))

    from avis.distance import DistanceEstimator
    e = DistanceEstimator(path="/tmp/_no_such_calibration.json")
    check("без калібрування метри ВСЕ ОДНО є (одна гілка коду)",
          e.meters(300) is not None)


# ── ФАЗА 8-10: конвеєр, статистика, оркестратор ──────────────────────────
def _stub_frame(w=960, h=720):
    import numpy as np
    return np.zeros((h, w, 3), np.uint8)


def _person(x, tid, y1=100, y2=400):
    from avis.models import Detection
    return Detection(id=tid, xyxy=(float(x), float(y1), float(x + 80), float(y2)),
                     conf=0.9, cls=0, name="person")


def _make_pipeline(events=None):
    from avis.control import ErrorCalculator
    from avis.control.follow import FollowController
    from avis.distance import DistanceEstimator
    from avis.perception import KalmanFilter, TargetSelector
    from avis.pipeline import TrackingPipeline
    return TrackingPipeline(
        detector=None, selector=TargetSelector(), tracker=KalmanFilter(),
        error_calculator=ErrorCalculator(estimator=DistanceEstimator()),
        controller=FollowController(),
        on_event=(events.append if events is not None else None))


def test_pipeline():
    print("\n[ TrackingPipeline — чиста обробка без IO ]")
    from avis.pipeline import LOST, PRED, VIS

    frame = _stub_frame()
    p = _make_pipeline()

    # без обраної цілі
    r = p.process(frame, 0.1, detections=[_person(400, 1)])
    check("ціль не обрано → LOST", r.status == LOST)
    check("LOST → команди немає", r.command is None)
    check("has_target=False", not r.has_target)

    # обрали й ведемо
    p.select_target(_person(400, 1))
    r = p.process(frame, 0.1, detections=[_person(400, 1)])
    check("ціль обрано → VIS", r.status == VIS)
    check("VIS → є команда", r.command is not None)
    check("VIS → has_target=True", r.has_target)
    check("центр обчислено", r.center is not None)

    # ціль зникла → PRED, і has_target МАЄ бути False
    r2 = p.process(frame, 0.1, detections=[])
    check("ціль зникла → PRED", r2.status == PRED)
    check("PRED → has_target=False (safety має спрацювати)", not r2.has_target)
    check("PRED → дистанцією не керуємо", r2.command is None or r2.command.forward == 0.0)

    # подія про підміну цілі
    events = []
    p2 = _make_pipeline(events)
    p2.select_target(_person(400, 1))
    p2.process(frame, 0.1, detections=[_person(400, 1)])
    p2.process(frame, 0.1, detections=[_person(410, 2)])   # інший id поруч
    check("підміна цілі повідомляється подією", any("змінилась" in e for e in events))


def test_pipeline_offframe():
    print("\n[ TrackingPipeline — прогноз затиснуто в кадр ]")
    frame = _stub_frame()
    p = _make_pipeline()
    p.select_target(_person(800, 1))
    # ведемо ціль, що швидко йде вправо за межу кадру
    x = 800
    for _ in range(6):
        x += 30
        p.process(frame, 0.1, detections=[_person(min(x, 870), 1)])
    r = p.process(frame, 0.1, detections=[])       # зникла за краєм
    check("прогноз не виходить за кадр",
          r.predicted is None or 0 <= r.predicted[0] <= 959)


def test_stats():
    print("\n[ RunStats ]")
    from avis.models import Command
    from avis.pipeline import FrameResult, VIS
    from avis.stats import RunStats

    st = RunStats()
    check("порожній звіт для 0 кадрів", st.report() == [])

    r = FrameResult(status=VIS, center=(480, 360),
                    command=Command(50, 0, 40))
    st.add(r, Command(30, 0, 20), 960, 720)      # обрізано по обох осях
    st.add(r, Command(30, 0, 20), 960, 720)
    lines = "\n".join(st.report())
    check("звіт містить кадри", "2 кадрів" in lines)
    check("сатурація помічена", "сатурація" in lines)
    check("дрож порахована", "дрож" in lines)


def test_dual_measure():
    print("\n[ Подвійний вимір дистанції: висота + ширина ]")
    from avis.distance import DistanceEstimator

    e = DistanceEstimator()
    W, H = 960, 720

    d1, m1 = e.meters_from_box((400, 150, 540, 540), W, H)   # рамка ціла
    check("рамка ціла → міряємо ВИСОТОЮ", m1 == "height" and d1 > 0)

    d2, m2 = e.meters_from_box((300, 0, 660, 719), W, H)     # зріст не вміщається
    check("зріст обрізано → переходимо на ШИРИНУ", m2 == "width" and d2 > 0)
    check("ширина дає БЛИЖЧУ дистанцію, ніж дала б висота", d2 < 1.0)

    d3, m3 = e.meters_from_box((0, 0, 959, 719), W, H)       # обрізано скрізь
    check("обрізано скрізь → виміру немає", d3 is None and m3 is None)

    # головне: 0.5 м тепер ВИМІРЮВАНІ
    w_at_05 = e.k_width / 0.5
    check("на 0.5 м ширина вміщається в кадр", w_at_05 <= W)
    check("на 0.5 м зріст НЕ вміщається (тому й потрібна ширина)",
          e.height_for(0.5) > H)


def test_proximity_guard():
    print("\n[ ЗАПОБІЖНИК БЛИЗЬКОСТІ — головний фікс 'летить в об'єкт' ]")
    from avis.control.follow.config import FollowConfig
    from avis.control.follow.distance import DistanceAxis

    cfg = FollowConfig()

    # Найважливіший сценарій: оцінка дистанції БРЕШЕ (людина повернулась боком,
    # плечі вужчі → "ціль далеко"), але рамка займає пів кадру.
    a = DistanceAxis(cfg)
    lying = a.command(_err(dist=3.0, fill=0.70), 0.1)   # "далеко" + рамка велика
    check("оцінка бреше 'далеко', але рамка велика → ВІДСТУП, не наступ", lying < 0)

    b = DistanceAxis(cfg)
    stop = b.command(_err(dist=3.0, fill=0.55), 0.1)
    check("рамка > порогу зупинки → вперед заборонено", stop <= 0)

    c = DistanceAxis(cfg)
    ok = c.command(_err(dist=3.0, fill=0.10), 0.1)
    check("рамка мала → рух уперед дозволено", ok > 0)

    # Запобіжник має бути ДІЄВИМ, а не символічним
    d = DistanceAxis(cfg)
    force = d.command(_err(dist=3.0, fill=0.9), 0.1)
    check("команда відступу вища за поріг чутливості дрона (20)", abs(force) >= 20)

    # Медіанний фільтр гасить одиничний викид від зміни пози
    e = DistanceAxis(FollowConfig(distance_median_frames=5))
    for _ in range(5):
        e.command(_err(dist=1.0), 0.1)
    before = e.smooth(1.0)
    spike = e.smooth(5.0)          # різкий хибний вимір
    check("медіана гасить одиничний викид", abs(spike - before) < 0.5)


def test_approach_converges():
    """Ключова властивість: дрон МАЄ дійти до уставки, а не спинитись раніше."""
    print("\n[ Підхід сходиться до уставки ]")
    from avis.control.flight import FlightSupervisor
    from avis.control.drone import NullDrone
    from avis.control.follow import FollowController
    from avis.distance import DistanceEstimator

    est = DistanceEstimator()
    sup = FlightSupervisor(NullDrone(), enable_forward=True)
    ctrl = FollowController()
    target = ctrl.distance_target_m

    def motor_cmd(d):
        h = min(635.0 / d, 720)
        w = est.k_width / d
        fill = est.frame_fill((0, 0, w, h), 960, 720)
        c = FollowController().update(_err(dist=d, fill=fill), 0.15)
        if c.forward == 0:
            return 0.0
        ceiling = sup._max_forward if c.forward >= 0 else sup._max_backward
        return sup._shape(c.forward, ceiling, floor=sup._min_forward)

    check("здалеку (2 м) — їде вперед", motor_cmd(2.0) > 0)
    check("на 1.2 м — ВСЕ ЩЕ їде (не глухне зарано)", motor_cmd(1.2) > 0)
    check("на 1.0 м — все ще їде", motor_cmd(1.0) > 0)
    check("на уставці — стоїть", motor_cmd(target) == 0)
    check("надто близько (0.45 м) — відступає", motor_cmd(0.45) < 0)


# ── Раннер ───────────────────────────────────────────────────────────────
PHASES = [test_config, test_axis, test_yaw_axis, test_vertical_axis,
          test_distance_axis, test_alignment_gate, test_controller,
          test_no_dual_setpoint, test_pipeline, test_pipeline_offframe,
          test_stats, test_dual_measure, test_proximity_guard,
          test_approach_converges, test_effectiveness]


def main():
    print("=" * 60)
    print("ТЕСТИ ПАКЕТА avis.control.follow")
    print("=" * 60)
    for t in PHASES:
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            _FAILURES.append(f"{t.__name__}: {type(e).__name__}: {e}")
            print(f"  ✗ EXCEPTION у {t.__name__}: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТ: {_PASS} пройдено, {_FAIL} провалено")
    for f in _FAILURES:
        print(f"  - {f}")
    print("=" * 60)
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
