# Джерело кадрів. ТУТ ІНТЕРФЕЙС ЗАЛИШАЄМО (на відміну від решти компонентів),
# бо підстановка реальна й близька: на roadmap — дрон Tello. Коли він з'явиться,
# додамо TelloSource(FrameSource) поряд, і App не зміниться ані на рядок.
#
# Це і є правило "золотої середини": інтерфейс тримаємо там, де він РЕАЛЬНО
# окупиться (≥2 реалізації), а не про запас.

from abc import ABC, abstractmethod

import cv2


class FrameSource(ABC):
    """Контракт джерела кадрів: звідки брати зображення."""

    @abstractmethod
    def read(self):
        """Повернути наступний BGR-кадр, або None якщо кадру немає."""

    @abstractmethod
    def release(self):
        """Звільнити ресурс (камеру/файл)."""


class WebcamSource(FrameSource):
    """Реалізація джерела на веб-камері (OpenCV)."""

    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)

    def read(self):
        ok, frame = self._cap.read()      # tuple unpacking: (успіх, кадр)
        return frame if ok else None

    def release(self):
        self._cap.release()
