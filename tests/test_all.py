# Регресійний тест-набір усього проєкту. Без залежностей (plain asserts),
# запуск:  .venv/bin/python tests/test_all.py
#
# Мета: після будь-якої зміни швидко переконатись, що нічого не зламалось —
# від value-об'єктів до повного конвеєра App і реплею.

import contextlib
import io
import os
import sys

import numpy as np

# Щоб `avis` імпортувався незалежно від того, звідки запущено (корінь проєкту).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- крихітний тест-харнес (замість pytest, щоб не тягти залежність) ---
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


@contextlib.contextmanager
def quiet():
    """Приглушити внутрішні print() ([flight]/[rec]) під час дії."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


from avis.models import Command, ControlError, Detection


def person(x, tid, name="person", cls=0):
    """Detection 'людини' в кадрі (для тестів трекінгу)."""
    return Detection(id=tid, xyxy=(float(x), 100.0, float(x + 60), 300.0),
                     conf=0.9, cls=cls, name=name)


def err(ex=0.0, ey=0.0, area=30000.0, w=960, h=720):
    return ControlError(error_x=ex, error_y=ey, area=area,
                        norm_x=ex / (w / 2), norm_y=ey / (h / 2))


# ─────────────────────────────────────────────────────────────────────────
def test_models():
    print("\n[ models / Detection ]")
    d = Detection(id=1, xyxy=(10, 20, 30, 60), conf=0.5, cls=0)
    check("center правильний", d.center == (20.0, 40.0))
    check("area правильна", d.area == 20 * 40)
    check("name default ''", d.name == "")
    c = Command(1, 2, 3)
    check("Command має 3 осі (yaw/vertical/forward)", (c.yaw, c.vertical, c.forward) == (1, 2, 3))


def test_error_calculator():
    print("\n[ ErrorCalculator ]")
    from avis.control import ErrorCalculator
    ec = ErrorCalculator()
    e = ec.compute(cx=640, cy=360, area=5000, frame_width=960, frame_height=720)
    check("error_x = cx - w/2", approx(e.error_x, 160))
    check("norm_x = ex/(w/2)", approx(e.norm_x, 160 / 480))
    check("area проходить як є", e.area == 5000)
    check("центр → помилка 0", ErrorCalculator().compute(480, 360, 1, 960, 720).error_x == 0)


def test_controller():
    print("\n[ FollowController (pure yaw, без строуфа) ]")
    from avis.control.controller import PID, FollowController
    p = PID(kp=0.2, ki=0, kd=0)
    o1 = p.update(100, 0.1)
    check("PID: стала помилка → стабільний вихід", approx(p.update(100, 0.1), o1))
    p.reset()
    check("PID.reset() очищає стан", p._prev_error is None and p._integral == 0)

    fc = FollowController(desired_area=60000)
    centered = fc.update(err(ex=0, area=30000), 1 / 30)
    check("центр: yaw≈0", abs(centered.yaw) < 1e-6)
    check("центр: forward>0 (їде до далекої цілі)", centered.forward > 0)
    # ГЕЙТ ВИРІВНЮВАННЯ розширено (align_full 0.35 / align_stop 0.80): дані
    # показали, що старий вузький гейт душив рух уперед у 100% кадрів, і дрон
    # ледве наздоганяв ціль. Тепер повне блокування — лише коли ціль майже
    # на краю кадру.
    mid = FollowController(desired_area=60000).update(err(ex=250, area=30000), 1 / 30)
    check("збоку помірно: yaw центрує", abs(mid.yaw) > 0)
    check("збоку помірно: forward ЧАСТКОВО дозволено", mid.forward > 0)

    edge = FollowController(desired_area=60000).update(err(ex=420, area=30000), 1 / 30)
    check("ціль на краю: forward повністю задушено", edge.forward == 0.0)

    far = FollowController(desired_area=60000).update(err(ex=0, area=20000), 1 / 30).forward
    near = FollowController(desired_area=60000).update(err(ex=0, area=100000), 1 / 30).forward
    check("асиметрія: назад ≈ 0.33×вперед", far > 0 and near < 0 and approx(abs(near) / far, 0.33, 0.05))

    dz = FollowController().update(err(ex=10, area=60000), 1 / 30)  # norm_x≈0.02 < deadzone
    check("deadzone: біля центру yaw=0", dz.yaw == 0.0)


def test_kalman():
    print("\n[ KalmanFilter ]")
    from avis.perception.tracking_filter import KalmanFilter
    kf = KalmanFilter()
    check("initialized False спочатку", not kf.initialized)
    check("predict() без стану → None", kf.predict(0.1) is None)
    x, y = kf.correct(100, 200)
    check("перший correct → вимір як є", (x, y) == (100, 200))
    check("initialized True після correct", kf.initialized)

    kf2 = KalmanFilter()
    kf2.correct(100, 240)
    kf2.predict(0.1); kf2.correct(120, 240)
    kf2.predict(0.1); kf2.correct(140, 240)
    px, _ = kf2.predict(0.1)
    check("predict екстраполює за швидкістю (x росте)", px > 140)

    steps = []
    prev = px
    for _ in range(60):
        nx, _ = kf2.predict(0.1)
        steps.append(nx - prev); prev = nx
    check("blind decay: пізніший крок менший за ранній", abs(steps[-1]) < abs(steps[0]))
    kf2.reset()
    check("reset() → initialized False", not kf2.initialized)


def test_selector():
    print("\n[ TargetSelector: оклюзія / перезахоплення / клік ]")
    from avis.perception.target_selector import TargetSelector

    s = TargetSelector()
    d = person(100, 7)
    got = s.select_at(85, 200, [d])           # 15px повз лівий край рамки
    check("select_at прощає неточне влучання", got is not None and got.id == 7)
    check("select_at далеко → None", TargetSelector().select_at(500, 450, [d]) is None)

    s2 = TargetSelector(); s2.select_target(person(100, 1))
    t, lost = s2.resolve([person(100, 1)])
    check("resolve за id повертає ціль", t is not None and not lost)

    # ОКЛЮЗІЯ з дебаунсом: перші кадри просідання ще НЕ вважаються втратою
    # (це найчастіше мерехтіння детектора), лише стійке просідання.
    s3 = TargetSelector(occlusion_debounce=3)
    s3.select_target(person(100, 1))               # еталонна площа велика
    small = Detection(id=1, xyxy=(100, 100, 110, 130), conf=.9, cls=0)  # крихітна
    t1, lost1 = s3.resolve([small])
    check("оклюзія, 1-й кадр: ще віримо рамці", t1 is not None and not lost1)
    s3.resolve([small])
    t3, lost3 = s3.resolve([small])
    check("оклюзія, 3-й кадр поспіль → (None, lost=True)", t3 is None and lost3)

    # одиничне мерехтіння НЕ має ламати трекінг
    s3b = TargetSelector(occlusion_debounce=3)
    s3b.select_target(person(100, 1))
    s3b.resolve([small])                            # блимнуло
    t_ok, lost_ok = s3b.resolve([person(100, 1)])   # знову нормальна рамка
    check("блимнуло і відновилось → трекінг цілий", t_ok is not None and not lost_ok)

    s4 = TargetSelector(); s4.select_target(person(100, 1))
    t, lost = s4.resolve([person(105, 9)], predicted_center=(135, 200))
    check("reacquire: новий id біля прогнозу", t is not None and s4.target_id == 9)

    s5 = TargetSelector(lost_grace_seconds=1.0); s5.select_target(person(100, 1))
    for _ in range(40):
        t, lost = s5.resolve([], predicted_center=(130, 200), dt=0.1)
    check("після пільги → (None, False) + очищено", t is None and not lost and s5.target_id is None)


def test_supervisor():
    print("\n[ FlightSupervisor ]")
    from avis.control.flight import FlightSupervisor, GROUNDED, HOVER, AUTO
    from avis.control.drone import NullDrone, SimDrone
    DT = 1 / 30
    C = Command(80, 60, 90)

    s = FlightSupervisor(NullDrone(), settle_after_takeoff=1.0)
    check("GROUNDED: update → None", s.update(C, True, DT) is None)
    with quiet():
        s.takeoff()
    check("takeoff → HOVER", s.state == HOVER)
    with quiet():
        s.arm()
    check("arm → AUTO", s.state == AUTO)
    early = s.update(C, True, DT)
    check("відстоювання: одразу після зльоту команди 0", early.yaw == 0 and early.forward == 0)
    with quiet():
        for _ in range(int(1.2 / DT)):
            late = s.update(C, True, DT)
    check("після відстоювання команди йдуть", late.yaw != 0)

    s2 = FlightSupervisor(NullDrone())
    with quiet():
        s2.takeoff(); s2.arm()
        for _ in range(int(10 / DT)):
            s2.update(None, False, DT)
    check("10с без цілі до захоплення → НЕ сідає", s2.state == AUTO)

    s3 = FlightSupervisor(NullDrone(), settle_after_takeoff=0.5)
    with quiet():
        s3.takeoff(); s3.arm()
        for _ in range(int(0.6 / DT)):
            s3.update(Command(10, 0, 0), True, DT)
        for _ in range(int(9 / DT)):
            s3.update(None, False, DT)
    check("захопили, тоді 9с втрати → GROUNDED", s3.state == GROUNDED)

    class LowBat(NullDrone):
        def battery(self):
            return 5
    s4 = FlightSupervisor(LowBat())
    with quiet():
        s4.takeoff(); s4.arm()
        for _ in range(int(2.5 / DT)):
            s4.update(Command(10, 0, 0), True, DT)
    check("заряд<15% → примусова посадка", s4.state == GROUNDED)

    s5 = FlightSupervisor(SimDrone(height=190), start_armed=True)
    check("висота OFF: вертикаль проходить", s5._apply_altitude(40) == 40)
    s6 = FlightSupervisor(SimDrone(height=190), start_armed=True, altitude_limits=True)
    check("висота ON: над стелею вгору=0", s6._apply_altitude(40) == 0)
    s7 = FlightSupervisor(SimDrone(height=40), start_armed=True, altitude_limits=True)
    check("висота ON: під підлогою вниз=0", s7._apply_altitude(-40) == 0)

    s8 = FlightSupervisor(NullDrone())
    check("_shape: велика → стеля", s8._shape(500, 70) == 70)
    check("_shape: слабка ненульова → min_speed", s8._shape(3, 70) == s8._min_speed)
    check("_shape: 0 → 0", s8._shape(0, 70) == 0)

    # ТРИ окремі стелі: горизонталь і поворот прискорені, ВЕРТИКАЛЬ недоторкана
    s9 = FlightSupervisor(NullDrone(), enable_forward=True)
    check("max_forward = 30 (відкат)", s9._max_forward == 30)
    check("max_speed = 70 (єдина стеля yaw+вертикаль)", s9._max_speed == 70)
    
    lim = s9._limit(Command(yaw=999, vertical=999, forward=999))
    check("_limit: forward ріжеться до 30", lim.forward == 30)
    check("_limit: yaw ріжеться до 70", lim.yaw == 70)
    check("_limit: vertical ріжеться до 70", lim.vertical == 70)

    # кеш заряду для логу
    s10 = FlightSupervisor(NullDrone())
    check("battery=None до зльоту", s10.battery is None)
    with quiet():
        s10.takeoff()
    check("battery заповнюється при зльоті", s10.battery == 100)


def test_drone():
    print("\n[ Drone (контракт) ]")
    from avis.control.drone import Drone, NullDrone, SimDrone, TelloDrone
    for cls in (NullDrone, SimDrone, TelloDrone):
        check(f"{cls.__name__} реалізує Drone", issubclass(cls, Drone))
    check("SimDrone.height() задається", SimDrone(height=123).height() == 123)
    check("NullDrone.battery()=100", NullDrone().battery() == 100)

    sent = {}
    class FakeTello:
        def send_rc_control(self, lr, fb, ud, yaw): sent.update(lr=lr, fb=fb, ud=ud, yaw=yaw)
    TelloDrone(FakeTello()).send(Command(yaw=5, vertical=6, forward=7))
    check("TelloDrone: канали мапляться (left_right=0)",
          sent["lr"] == 0 and sent["fb"] == 7 and sent["ud"] == 6 and sent["yaw"] == 5)


def test_bugfixes():
    print("\n[ Виправлені баги (регресійні запобіжники) ]")
    from avis.perception.target_selector import TargetSelector
    from avis.control.controller import FollowController
    from avis.control.flight import FlightSupervisor, GROUNDED, HOVER, AUTO, BUSY
    from avis.control.drone import Drone, NullDrone

    # БАГ #1: у PRED supervisor мусить бачити has_target=False і зупинити рух
    class Spy(Drone):
        def __init__(s): s.sent = []
        def takeoff(s): pass
        def land(s): s.sent.append("LAND")
        def emergency(s): pass
        def send(s, c): s.sent.append((round(c.yaw), round(c.vertical), round(c.forward)))
        def battery(s): return 90
        def height(s): return 100

    DT = 1 / 10
    d = Spy(); sup = FlightSupervisor(d, enable_forward=True, settle_after_takeoff=0)
    with quiet():
        sup.takeoff()
        while sup.state == BUSY:
            pass
        sup.arm()
    d.sent.clear()
    with quiet():
        for _ in range(int(3 / DT)):
            sup.update(Command(100, 0, 100), False, DT)   # ЦІЛІ НЕМА
    nz = [x for x in d.sent if isinstance(x, tuple) and any(x)]
    check("#1 без цілі → зупиняється (не крутиться за привидом)", len(nz) <= 6)

    # БАГ #2: клас цілі НЕ перезаписується, навіть якщо YOLO змінила клас
    s = TargetSelector()
    s.select_target(person(100, 1))                        # обрали ЛЮДИНУ (cls=0)
    chair_same_id = Detection(id=1, xyxy=(100, 100, 160, 300), conf=.9, cls=56, name="chair")
    s.resolve([chair_same_id])                             # YOLO раптом каже "стілець"
    check("#2 клас цілі лишився людиною", s._target_cls == 0)

    # БАГ #3: без довіри до дистанції рух уперед = 0
    fc = FollowController(desired_area=60000)
    c_far = fc.update(err(ex=0, area=10000), 1 / 30, trust_distance=True)
    c_pred = FollowController(desired_area=60000).update(
        err(ex=0, area=10000), 1 / 30, trust_distance=False)
    check("#3 VIS: їде вперед", c_far.forward > 0)
    check("#3 PRED: forward=0 (не летить на застарілій площі)", c_pred.forward == 0.0)

    # БАГ #6: клік між двох рамок бере ту, чий ЦЕНТР ближчий
    s2 = TargetSelector()
    big = Detection(id=1, xyxy=(100, 100, 300, 500), conf=.9, cls=0, name="person")
    small = Detection(id=2, xyxy=(180, 380, 240, 440), conf=.9, cls=56, name="chair")
    picked = s2.select_at(200, 200, [big, small])          # клік ближче до центру ЛЮДИНИ
    check("#6 клік бере найближчу за центром, а не найменшу", picked.id == 1)

    # БАГ #9: радіус перезахоплення зменшено
    check("#9 стеля радіуса перезахоплення ≤ 250px", TargetSelector()._reacquire_radius_max <= 250)

    # БАГ #12: команди округлюються, а не обрізаються
    from avis.control.drone import TelloDrone
    sent = {}
    class FakeTello:
        def send_rc_control(self, lr, fb, ud, yaw): sent.update(fb=fb, ud=ud, yaw=yaw)
    TelloDrone(FakeTello()).send(Command(yaw=12.9, vertical=-12.9, forward=12.4))
    check("#12 round(), а не int()", sent["yaw"] == 13 and sent["ud"] == -13 and sent["fb"] == 12)


def test_arm_and_backward():
    print("\n[ G під час зльоту + повільний відступ ]")
    import time as _t
    from avis.control.drone import Drone, NullDrone
    from avis.control.flight import FlightSupervisor, AUTO, HOVER, BUSY

    # G, натиснута ПІД ЧАС зльоту, не має пропадати
    class SlowDrone(Drone):
        def takeoff(s): _t.sleep(1.0)
        def land(s): pass
        def emergency(s): pass
        def send(s, c): pass
        def battery(s): return 90
        def height(s): return 100

    sup = FlightSupervisor(SlowDrone())
    with quiet():
        sup.takeoff()
        sup.toggle()                       # G ще під час зльоту
    check("G під час зльоту: стан ще BUSY", sup.state == BUSY)
    _t.sleep(1.4)
    check("після зльоту стеження увімкнулось САМО", sup.state == AUTO)

    # повторна G має вимикати (а не вмикати знову)
    with quiet():
        sup.toggle()
    check("повторна G вимикає стеження", sup.state == HOVER)

    # відступ удвічі повільніший за наступ
    s2 = FlightSupervisor(NullDrone(), enable_forward=True)
    fwd = s2._limit(Command(0, 0, 999)).forward
    bwd = s2._limit(Command(0, 0, -999)).forward
    check("вперед стеля 30", fwd == 30)
    check("назад стеля 15 (удвічі повільніше)", bwd == -15)


def test_key_debounce():
    print("\n[ Дебаунс клавіш (одне натискання = одна дія) ]")
    from avis.app import App
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor, AUTO, HOVER
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator
    from avis.control.controller import FollowController
    from avis.view import HeadlessRenderer

    sup = FlightSupervisor(NullDrone())
    app = App(source=None, detector=None, selector=TargetSelector(),
              tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
              controller=FollowController(), renderer=HeadlessRenderer(),
              supervisor=sup)
    with quiet():
        sup.takeoff()
        while sup.state == "BUSY":
            pass
        # ДВІ однакові події поспіль (дубль від вікна + терміналу)
        app._handle_key(ord('g'))
        app._handle_key(ord('g'))
    check("дубль G не скасовує сам себе — лишається AUTO", sup.state == AUTO)


def test_nonblocking_takeoff():
    print("\n[ Зліт не блокує цикл (відео не зависає) ]")
    import time as _t
    from avis.control.drone import Drone
    from avis.control.flight import FlightSupervisor, BUSY, HOVER

    class SlowDrone(Drone):
        """Імітує Tello: takeoff відповідає аж через 2 с (реально буває до 20)."""
        def takeoff(s): _t.sleep(2.0)
        def land(s): pass
        def emergency(s): pass
        def send(s, c): pass
        def battery(s): return 90
        def height(s): return 100

    sup = FlightSupervisor(SlowDrone())
    t0 = _t.time()
    with quiet():
        sup.takeoff()
    elapsed = _t.time() - t0
    check(f"takeoff() повертає керування миттєво ({elapsed:.2f}с)", elapsed < 0.3)
    check("під час зльоту стан BUSY", sup.state == BUSY)
    check("під час зльоту в мотори нічого не шлемо", sup.update(Command(50, 0, 0), True, 0.1) is None)
    _t.sleep(2.3)
    check("після завершення стан HOVER", sup.state == HOVER)


def test_key_handling():
    print("\n[ Клавіші: коди з високими бітами (macOS/Qt) ]")
    from avis.app import App
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor, GROUNDED, HOVER, AUTO
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator, FollowController
    from avis.view.renderer import Renderer

    class Head(Renderer):
        def set_mouse_callback(s, c): pass
        def draw(s, *a, **k): pass
        def is_open(s): return True

    def make():
        sup = FlightSupervisor(NullDrone())
        app = App(source=None, detector=None, selector=TargetSelector(),
                  tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
                  controller=FollowController(), renderer=Head(), supervisor=sup)
        return app, sup

    HIGH = 0x100000        # старші біти, які додає macOS/Qt

    # чистий ASCII працює
    app, sup = make()
    with quiet():
        app._handle_key(ord('t'))
    check("чистий код 't' → зліт", sup.state == HOVER)
    with quiet():
        app._handle_key(ord('g'))
    check("чистий код 'g' → стеження", sup.state == AUTO)

    # РЕГРЕСІЯ, яку ловимо: код із високими бітами теж має спрацьовувати
    app2, sup2 = make()
    with quiet():
        app2._handle_key(HIGH | ord('t'))
    check("код 't' З ВИСОКИМИ БІТАМИ → зліт", sup2.state == HOVER)
    with quiet():
        app2._handle_key(HIGH | ord('g'))
    check("код 'g' З ВИСОКИМИ БІТАМИ → стеження", sup2.state == AUTO)
    with quiet():
        app2._handle_key(HIGH | ord('l'))
    check("код 'l' З ВИСОКИМИ БІТАМИ → посадка", sup2.state == GROUNDED)

    # вихід теж має ловитись в обох формах
    app3, _ = make()
    check("вихід: чистий Esc", app3._handle_key(27) is True)
    check("вихід: 'q' з високими бітами", app3._handle_key(HIGH | ord('q')) is True)


def test_integration_live():
    print("\n[ Інтеграція: повний App (живий стаб) ]")
    from avis.app import App
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator, FollowController
    from avis.view.renderer import Renderer

    class LiveSource:
        def __init__(s, n=6): s._n = n
        def read(s):
            if s._n <= 0: return None
            s._n -= 1
            return np.zeros((480, 640, 3), np.uint8)
        def release(s): pass

    class StubDet:
        def __init__(s): s.c = 0
        def detect(s, f):
            s.c += 1
            return [Detection(id=1, xyxy=(450, 220, 510, 340), conf=.9, cls=0)]

    class Head(Renderer):
        def set_mouse_callback(s, c): pass
        def draw(s, *a, **k): pass
        def is_open(s): return True

    det = StubDet()
    app = App(source=LiveSource(), detector=det, selector=TargetSelector(),
              tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
              controller=FollowController(), renderer=Head(),
              supervisor=FlightSupervisor(NullDrone(), start_armed=True, enable_forward=True))
    app._selector.select_target(Detection(id=1, xyxy=(450, 220, 510, 340), conf=.9, cls=0))
    with quiet():
        app.run()
    check("App.run() без крашів", True)
    check("детектор викликався щокадру", det.c == 6)





def test_integration_replay():
    print("\n[ Інтеграція: запис → реплей ]")
    from avis.app import App
    from avis.replay import FlightRecorder, ReplaySource
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.perception import KalmanFilter, TargetSelector
    from avis.control import ErrorCalculator, FollowController
    from avis.view.renderer import Renderer

    name = "_pytest_tmp"
    with quiet():
        rec = FlightRecorder(name)
        x = 200
        for i in range(20):
            x += 12
            frame = np.zeros((480, 640, 3), np.uint8)
            dets = [Detection(id=1, xyxy=(x, 220, x + 50, 300), conf=.9, cls=0, name="person")]
            # Пишемо і базову лінію: що пішло в мотори, стан, заряд.
            rec.write(frame, 1 / 30, dets, 1,
                      sent=Command(yaw=5, vertical=-2, forward=30),
                      state="AUTO", battery=100)
        rec.close()

    rs = ReplaySource(name)
    f = rs.read()
    check("ReplaySource.read() дає кадр", f is not None and f.shape == (480, 640, 3))
    check("replayed_dt збережений", approx(rs.replayed_dt(), 1 / 30, 1e-3))
    check("replayed_detections відновлює Detection", len(rs.replayed_detections()) == 1)
    base = rs.baseline()
    check("baseline: команди того польоту збережені",
          base["sent"] is not None and base["sent"]["forward"] == 30.0)
    check("baseline: стан і заряд збережені",
          base["state"] == "AUTO" and base["battery"] == 100)
    rs.release()

    class Head(Renderer):
        def set_mouse_callback(s, c): pass
        def draw(s, *a, **k): pass
        def is_open(s): return True

    with quiet():
        app = App(source=ReplaySource(name), detector=None, selector=TargetSelector(),
                  tracker=KalmanFilter(), error_calculator=ErrorCalculator(),
                  controller=FollowController(), renderer=Head(),
                  supervisor=FlightSupervisor(NullDrone(), start_armed=True, enable_forward=True))
        app.run()
    check("App у режимі реплею відпрацював", app._replaying)

    for ext in (".mp4", ".jsonl"):
        p = os.path.join("recordings", name + ext)
        if os.path.exists(p):
            os.remove(p)
    check("тимчасові файли реплею прибрано", True)


def main():
    print("=" * 60)
    print("РЕГРЕСІЙНИЙ ТЕСТ-НАБІР AVIS")
    print("=" * 60)
    for t in (test_models, test_error_calculator, test_controller, test_kalman,
              test_selector, test_supervisor, test_drone, test_key_handling,
              test_bugfixes, test_nonblocking_takeoff,
              test_arm_and_backward, test_key_debounce,
              test_integration_live, test_integration_replay):
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            _FAILURES.append(f"{t.__name__} КИНУВ ВИНЯТОК: {type(e).__name__}: {e}")
            print(f"  ✗ EXCEPTION у {t.__name__}: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТ: {_PASS} пройдено, {_FAIL} провалено")
    if _FAILURES:
        print("Провали:")
        for f in _FAILURES:
            print(f"  - {f}")
    print("=" * 60)
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
