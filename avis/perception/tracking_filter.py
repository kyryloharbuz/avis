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
    def __init__(self, process_var=50.0, measurement_var=4.0):
        self._process_var = process_var
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

    # PREDICT: посунути стан на dt вперед; повернути (x, y). @ = множення матриць.
    def predict(self, dt):
        if self._x is None:
            return None
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(dt)
        return float(self._x[0]), float(self._x[1])

    # UPDATE: свіже вимірювання (cx, cy) → уточнити; повернути згладжений (x, y).
    def update(self, cx, cy, dt):
        if self._x is None:
            self._init_state(cx, cy)
            return cx, cy

        self.predict(dt)                                 # 1) передбачити
        z = np.array([[cx], [cy]], dtype=float)          # 2) корекція за виміром
        y = z - self._H @ self._x                        # innovation: факт - прогноз
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)       # Kalman gain: кому вірити
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return float(self._x[0]), float(self._x[1])
