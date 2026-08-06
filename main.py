# Точка входу — файл, який ти запускаєш: `python main.py`.
#
# COMPOSITION ROOT ("корінь композиції") — ЄДИНЕ місце, що знає конкретні
# реалізації й збирає з них систему, впроваджуючи їх у App через конструктор.

from datetime import datetime

# Імпортуємо коротко — через фасади підпакетів (їхні __init__.py).
from avis.app import App
from avis.control import ErrorCalculator, FollowController
from avis.perception import Detector, KalmanFilter, TargetSelector, WebcamSource
from avis.view import CvRenderer


def _build_tello(fly: bool, forward: bool):
    """Створити джерело кадрів і (за потреби) керування дроном.

    КЛЮЧОВА ДЕТАЛЬ: об'єкт Tello створюється ОДИН раз і передається і у
    відео, і в команди. Два незалежні Tello() конфліктували б за UDP-порт.

    Імпорт усередині функції навмисний: djitellopy+av важать ~22 МБ, і при
    роботі з вебкамерою вантажити їх немає сенсу."""
    from djitellopy import Tello

    from avis.control.drone import TelloDrone
    from avis.control.flight import FlightSupervisor
    from avis.perception.tello_source import TelloSource

    tello = Tello()
    source = TelloSource(tello=tello)          # той самий екземпляр ↓
    supervisor = None
    if fly:
        supervisor = FlightSupervisor(TelloDrone(tello), enable_forward=forward)
    return source, supervisor


def _build_replay(name: str):
    """Зібрати РЕПЛЕЙ: джерело з запису + supervisor на заглушці-дроні, одразу
    "у польоті", щоб бачити повний ланцюг команд аж до моторів. Детектора немає
    (детекції беруться з запису) — YOLO не вантажимо, тому реплей блискавичний."""
    from avis.control.drone import NullDrone
    from avis.control.flight import FlightSupervisor
    from avis.replay import ReplaySource

    source = ReplaySource(name)
    supervisor = FlightSupervisor(NullDrone(), enable_forward=True, start_armed=True)
    return source, supervisor


# Спеціальне значення: запис явно вимкнено прапорцем --no-record.
_NO_RECORD = object()


def build_app(use_tello=False, fly=False, forward=False,
              record_name=None, replay_name=None, person_only=False) -> App:
    """Зібрати додаток. Джерело кадрів обирається тут — у цьому вся суть
    інтерфейсу FrameSource (вебкамера / Tello / реплей взаємозамінні)."""
    # person_only=True → детектуємо ЛИШЕ людей (клас COCO 0). Найнадійніший
    # режим для follow-me: стільці й меблі тоді взагалі не можуть стати ціллю.
    detector = Detector(classes=[0] if person_only else None)
    recorder = None

    if replay_name:
        source, supervisor = _build_replay(replay_name)
        detector = None            # у реплеї детекції з запису, YOLO не треба
    elif use_tello:
        source, supervisor = _build_tello(fly, forward)
    else:
        source, supervisor = WebcamSource(), None

    # ЗАПИС. Ведеться на будь-якому ЖИВОМУ джерелі (вебкамера чи дрон) і
    # ВМИКАЄТЬСЯ САМ: наперед не вгадаєш, у якому польоті трапиться цікавий баг,
    # а без запису його вже не відтвориш. Ім'я за замовчуванням — таймстамп.
    if not replay_name and record_name != _NO_RECORD:
        from avis.replay import FlightRecorder
        if not record_name:
            record_name = "flight_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder = FlightRecorder(record_name)

    return App(
        source=source,
        supervisor=supervisor,
        detector=detector,
        selector=TargetSelector(),
        tracker=KalmanFilter(),
        error_calculator=ErrorCalculator(),
        controller=FollowController(),
        renderer=CvRenderer(),
        recorder=recorder,
    )


def _arg_value(argv, flag):
    """Витягти значення після прапорця: '--replay flight_001' → 'flight_001'."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


if __name__ == "__main__":
    # Запуск:
    #   python main.py                         — вебкамера
    #   python main.py --tello                 — відео з дрона, без польоту
    #   python main.py --tello --fly           — автономне стеження
    #   python main.py --replay flight_001     — ПРОГРАТИ запис (тюнінг без дрона)
    #
    # ЗАПИС УВІМКНЕНО ЗАВЖДИ (ім'я = таймстамп). Своє ім'я: --record my_test
    # Вимкнути: --no-record
    import sys
    if "--no-record" in sys.argv:
        rec = _NO_RECORD
    else:
        rec = _arg_value(sys.argv, "--record")
    build_app(
        use_tello="--tello" in sys.argv,
        fly="--fly" in sys.argv,
        forward="--no-forward" not in sys.argv,
        record_name=rec,
        replay_name=_arg_value(sys.argv, "--replay"),
        person_only="--person" in sys.argv,   # детектувати ЛИШЕ людей
    ).run()
