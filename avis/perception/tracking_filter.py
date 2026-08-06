# Фільтр Калмана (модель сталої швидкості) для згладжування/передбачення
# позиції цілі. Звичайний клас без інтерфейсу.
#
# Навіщо:
#   1) рамка детектора "тремтить" → центр стрибає → керування смикається;
#      Калман згладжує;
#   2) коли ціль на мить зникла — метод predict() ПЕРЕДБАЧАЄ, де вона, за
#      минулою швидкістю (рятує під час "TARGET LOST").
#
# Стан = [x, y, vx, vy] (позиція + швидкість). Вимірюємо лише (x, y).

import numpy as np


class KalmanFilter:
    # process_var — "шум процесу" (як сильно ціль може непередбачувано міняти
    #   швидкість); measurement_var — "шум вимірювання" (наскільки не довіряти
    #   детектору). Головні ручки налаштування.
    # blind_decay_after — скільки секунд "наосліп" (без вимірів) ще довіряємо
    #   сталій швидкості; blind_decay_per_second — до якої частки згасає
    #   швидкість за секунду після цього (0.05 = лишається 5%).
    # НАВІЩО: модель сталої швидкості чесна лише на короткій дистанції. Без
    #   згасання прогноз екстраполює ВІЧНО і "відлітає" за межі кадру (x=3000),
    #   через що PID б'є по фантому, а повторне захоплення шукає ціль там, де її
    #   бути не може. Зі згасанням прогноз плавно зупиняється там, де ціль зникла.
    def __init__(self, process_var=50.0, measurement_var=4.0,
                 blind_decay_after=0.4, blind_decay_per_second=0.05):
        self._process_var = process_var
        self._blind_decay_after = blind_decay_after
        self._blind_decay_per_second = blind_decay_per_second
        self._time_since_correct = 0.0   # секунд від останнього реального виміру
        self._H = np.array([[1, 0, 0, 0],       # H: міряємо лише x та y
                            [0, 1, 0, 0]], dtype=float)
        self._R = np.eye(2) * measurement_var   # R: шум вимірювання
        self._x = None  # стан 4x1 [[x],[y],[vx],[vy]] (None = не ініціалізовано)
        self._P = None  # коваріація 4x4 (наша невпевненість)

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def reset(self):
        self._x = None
        self._P = None
        self._time_since_correct = 0.0

    def _transition(self, dt):                  # F: x += vx*dt, y += vy*dt
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

    def _process_noise(self, dt):               # Q: шум процесу (стала швидкість)
        q = self._process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        return q * np.array([
            [dt3 / 3, 0,       dt2 / 2, 0],
            [0,       dt3 / 3, 0,       dt2 / 2],
            [dt2 / 2, 0,       dt,      0],
            [0,       dt2 / 2, 0,       dt],
        ], dtype=float)

    def _init_state(self, cx, cy):              # перше вимірювання: позиція з датчика
        self._x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
        self._P = np.eye(4) * 1000.0            # велика невпевненість спочатку

    # Класичний цикл Калмана — два ОКРЕМІ кроки: PREDICT, потім (якщо є вимір)
    # CORRECT. Раніше вони були злиті в update(); розділили, бо App робить
    # прогноз ОДИН раз на кадр і використовує його ще й для повторного
    # захоплення цілі (re-acquire); подвійне передбачення тут було б помилкою.

    # PREDICT: посунути стан на dt вперед; повернути (x, y). @ = множення матриць.
    # Виклик БЕЗ виміру (коли ціль перекрита/зникла) — стан "їде" далі за минулою
    # швидкістю. Це і є пам'ять про рух.
    def predict(self, dt):
        if self._x is None:
            return None
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(dt)

        # ЗГАСАННЯ ШВИДКОСТІ "наосліп": поки вимірів немає довше за
        # blind_decay_after — плавно гасимо vx, vy. Так прогноз не відлітає у
        # безкінечність, а зупиняється приблизно там, де ціль востаннє бачили.
        self._time_since_correct += dt
        if self._time_since_correct > self._blind_decay_after:
            factor = self._blind_decay_per_second ** dt   # напр. 0.05^dt за кадр
            self._x[2] *= factor                          # vx
            self._x[3] *= factor                          # vy
        return float(self._x[0]), float(self._x[1])

    # CORRECT: уточнити стан свіжим виміром (cx, cy); повернути згладжений (x, y).
    # Викликати ПІСЛЯ predict(). Перший вимір просто ініціалізує фільтр.
    def correct(self, cx, cy):
        self._time_since_correct = 0.0   # є реальний вимір → згасання скидаємо
        if self._x is None:
            self._init_state(cx, cy)
            return cx, cy
        z = np.array([[cx], [cy]], dtype=float)          # вимір (2x1)
        y = z - self._H @ self._x                        # innovation: факт - прогноз
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)       # Kalman gain: кому вірити
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return float(self._x[0]), float(self._x[1])
