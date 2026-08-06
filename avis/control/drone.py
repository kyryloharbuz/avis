# Виконавчий пристрій: те, що ПРИЙМАЄ команди керування.
#
# Інтерфейс лишаємо (за нашим правилом "≥2 реалізації"): їх одразу дві —
# справжній Tello і NullDrone, який лише друкує. Друга критично потрібна, щоб
# відлагоджувати автономну логіку БЕЗ ризику розбити дрон.

from abc import ABC, abstractmethod

from avis.models import Command


class Drone(ABC):
    """Контракт дрона: злетіти, сісти, прийняти команду швидкостей."""

    @abstractmethod
    def takeoff(self):
        """Злетіти (блокує до набору висоти)."""

    @abstractmethod
    def land(self):
        """М'яка посадка."""

    @abstractmethod
    def emergency(self):
        """АВАРІЯ: миттєво вимкнути мотори (дрон падає)."""

    @abstractmethod
    def send(self, command: Command):
        """Надіслати швидкості руху. Нулі = висіти на місці."""

    @abstractmethod
    def battery(self) -> int:
        """Заряд у відсотках."""

    @abstractmethod
    def height(self) -> int:
        """Поточна висота в САНТИМЕТРАХ (для обмежень стелі/підлоги)."""


class NullDrone(Drone):
    """Заглушка: нічого не літає, лише друкує. Для роботи з вебкамерою
    й для перевірки автономної логіки без ризику."""

    def __init__(self):
        self._height = 100.0     # см — симульована висота (стартуємо на 1 м)

    def takeoff(self):
        print("[drone] TAKEOFF (симуляція)")

    def land(self):
        print("[drone] LAND (симуляція)")

    def emergency(self):
        print("[drone] EMERGENCY (симуляція)")

    def send(self, command: Command):
        # Легко "інтегруємо" висоту з вертикальної швидкості, щоб обмеження
        # стелі/підлоги можна було перевірити тестами БЕЗ реального дрона.
        self._height += command.vertical * 0.05

    def battery(self) -> int:
        return 100

    def height(self) -> int:
        return self._height


class SimDrone(NullDrone):
    """NullDrone із заданою висотою — зручно для тестів обмежень висоти."""

    def __init__(self, height=100.0):
        super().__init__()
        self._height = height


class TelloDrone(Drone):
    """Справжній Tello.

    ВАЖЛИВО: приймає ГОТОВИЙ об'єкт Tello, а не створює свій. Причина — і
    відео, і команди йдуть через один UDP-сокет: два незалежні об'єкти Tello
    конфліктували б за порт. Тому в main.py дрон створюється ОДИН раз і
    передається і в TelloSource, і сюди."""

    def __init__(self, tello):
        self._tello = tello

    def takeoff(self):
        self._tello.takeoff()

    def land(self):
        self._tello.send_rc_control(0, 0, 0, 0)   # спершу зупинити рух
        self._tello.land()

    def emergency(self):
        self._tello.emergency()

    def send(self, command: Command):
        # Наш Command → чотири канали Tello. left_right не задіяний: доводимо
        # ціль до центру поворотом (yaw), як робить людина-оператор.
        # round(), а не int(): int обрізає до нуля (12.9 → 12), тобто систематично
        # з'їдав до одиниці швидкості на кожній осі. round() дає чесне значення.
        self._tello.send_rc_control(
            0,                          # left_right (боковий зсув) — не задіяний
            round(command.forward),     # forward_backward — дистанція
            round(command.vertical),    # up_down — висота
            round(command.yaw),         # yaw — поворот
        )

    def battery(self) -> int:
        return self._tello.get_battery()

    def height(self) -> int:
        # get_height() читає КЕШОВАНИЙ стан (Tello шле його безперервно окремим
        # потоком), тож це не UDP-запит і викликати щокадру дешево. У см.
        return self._tello.get_height()
