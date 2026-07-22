# Точка входу — файл, який ти запускаєш: `python main.py`.
#
# COMPOSITION ROOT ("корінь композиції") — ЄДИНЕ місце, що знає конкретні
# реалізації й збирає з них систему, впроваджуючи їх у App через конструктор.
#
# Завдяки цьому, щоб перевести проєкт на дрон Tello, достатньо створити
# TelloSource(FrameSource) і замінити ОДИН рядок нижче (source=...). App і всі
# компоненти чіпати не треба.

# Імпортуємо коротко — через фасади підпакетів (їхні __init__.py).
from avis.app import App
from avis.control import ErrorCalculator, FollowController
from avis.perception import Detector, KalmanFilter, TargetSelector, WebcamSource
from avis.view import CvRenderer


def build_app() -> App:
    """Зібрати додаток із конкретних реалізацій. Хочеш іншу поведінку —
    підстав інший клас саме тут."""
    return App(
        source=WebcamSource(),
        detector=Detector(),
        selector=TargetSelector(),
        tracker=KalmanFilter(),
        error_calculator=ErrorCalculator(),
        controller=FollowController(),
        renderer=CvRenderer(),
    )


if __name__ == "__main__":
    build_app().run()
