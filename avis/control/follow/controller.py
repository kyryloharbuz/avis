# КООРДИНАТОР стеження.
#
# Раніше цей клас був на 250 рядків і 20 параметрів, і поєднував пʼять
# відповідальностей: горизонталь, вертикаль, дистанцію (у ДВОХ несумісних
# стратегіях), зони спокою, адаптацію рівня, гейти й розвʼязку осей.
# Кожна зміна там ламала щось інше — це підтверджено практикою.
#
# Тепер він робить рівно одне: КООРДИНУЄ незалежні осі. Уся математика живе
# в них, а тут лишились тільки порядок виклику й звʼязок між осями:
#   • дистанція віддає ШВИДКІСТЬ своєї зміни → вертикаль вирішує, чи вірити собі;
#   • поворот визначає ВИРІВНЮВАННЯ → воно дозволяє чи душить рух уперед.

from avis.control.follow.alignment import AlignmentGate
from avis.control.follow.config import FollowConfig
from avis.control.follow.distance import DistanceAxis
from avis.control.follow.vertical import VerticalAxis
from avis.control.follow.yaw import YawAxis
from avis.models import Command, ControlError


class FollowController:
    """Зводить три незалежні осі в одну команду руху."""

    def __init__(self, config: FollowConfig = None, **overrides):
        # Дозволяємо і готовий конфіг, і точкові правки: FollowController(yaw_kp=0.5).
        # Друге зручно для тюнінгу на реплеї, коли перебираєш один параметр.
        self.config = config or FollowConfig(**overrides)
        self._yaw = YawAxis(self.config)
        self._vertical = VerticalAxis(self.config)
        self._distance = DistanceAxis(self.config)
        self._gate = AlignmentGate(self.config)

    def reset(self):
        """Скидання при зміні/втраті цілі: усі осі забувають накопичений стан."""
        self._yaw.reset()
        self._vertical.reset()
        self._distance.reset()

    def update(self, error: ControlError, dt, trust_distance=True) -> Command:
        """trust_distance=False у режимі PRED: свіжого виміру немає, керуємо
        лише напрямком. Раніше в цьому режимі використовувалась ЗАСТАРІЛА
        площа — і дрон летів уперед на людину, вважаючи її далекою."""
        # Швидкість зміни дистанції потрібна вертикалі — рахуємо ЗАВЖДИ,
        # інакше похідна брехатиме після кожної паузи.
        self._distance.track_rate(error.distance_m, dt)

        yaw = self._yaw.command(error, dt)
        vertical = self._vertical.command(error, dt, self._distance.rate)

        if not trust_distance:
            forward = self._distance.hold()
        else:
            # Рух уперед дозволено тою мірою, якою дрон уже дивиться на ціль.
            forward = self._distance.command(error, dt) * self._gate.factor(error.norm_x)

        return Command(yaw=yaw, vertical=vertical, forward=forward)

    # ── Для метрик і звітів ────────────────────────────────────────────
    @property
    def distance_target_m(self):
        """Уставка дистанції. ЄДИНЕ джерело істини — метрика бере її звідси,
        щоб міряти те саме, чим керує система (раніше вони розходились)."""
        return self.config.distance_target_m

    @property
    def distance_deadzone_m(self):
        return self.config.distance_deadzone_m
