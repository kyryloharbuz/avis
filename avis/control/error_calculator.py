# Обчислення помилки керування: наскільки ціль зміщена від центру кадру.
# Звичайний клас без інтерфейсу. Тримаємо його об'єктом (а не голою функцією),
# щоб App міг отримати його через впровадження залежностей (DI), однаково з рештою.

from avis.models import ControlError


class ErrorCalculator:
    # Приймаємо вже готові cx, cy (згладжений Калманом центр) і площу.
    def compute(self, cx, cy, area, frame_width, frame_height) -> ControlError:
        return ControlError(
            error_x=cx - frame_width / 2,    # правіше центру → +, лівіше → -
            error_y=cy - frame_height / 2,   # нижче центру → +, вище → -
            area=area,                       # площу передаємо як є
        )
