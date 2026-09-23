# Обчислення помилки керування: наскільки ціль зміщена від центру кадру.
# Звичайний клас без інтерфейсу. Тримаємо його об'єктом (а не голою функцією),
# щоб App міг отримати його через впровадження залежностей (DI), однаково з рештою.

from avis.models import ControlError


class ErrorCalculator:
    # estimator — оцінювач дистанції (може бути None). Якщо камеру
    # відкалібровано, помилка нестиме ще й відстань У МЕТРАХ.
    def __init__(self, estimator=None):
        self._estimator = estimator

    # Приймаємо вже готові cx, cy (згладжений Калманом центр) і площу.
    # box — рамка цілі (xyxy), потрібна для оцінки дистанції й обрізання.
    def compute(self, cx, cy, area, frame_width, frame_height, box=None) -> ControlError:
        ex = cx - frame_width / 2            # правіше центру → +, лівіше → -
        ey = cy - frame_height / 2           # нижче центру → +, вище → -

        distance_m, clipped, fill = None, False, 0.0
        if box is not None and self._estimator is not None:
            fill = self._estimator.frame_fill(box, frame_width, frame_height)
            # Естіматор САМ обирає міру: висоту, доки рамка ціла, і ширину,
            # коли зріст уже не вміщається в кадр. Завдяки цьому дистанція
            # лишається вимірюваною аж до ~0.3 м, а не пропадає на 0.88 м.
            distance_m, _measure = self._estimator.meters_from_box(
                box, frame_width, frame_height)
            # "Обрізано" тепер означає рівно одне: дистанцію ВИМІРЯТИ НЕМОЖЛИВО
            # (обидві міри вперлись у край). Раніше сюди потрапляло й звичайне
            # вертикальне обрізання — і рух уперед блокувався дарма.
            clipped = distance_m is None

        return ControlError(
            error_x=ex,
            error_y=ey,
            area=area,                       # площу передаємо як є
            # Нормування на ПІВширину: на краю кадру вийде рівно ±1.
            norm_x=ex / (frame_width / 2),
            norm_y=ey / (frame_height / 2),
            distance_m=distance_m,
            box_clipped=clipped,
            frame_fill=fill,
        )
