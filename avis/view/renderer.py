# Візуалізація: показ кадру з рамками/підписами й обробка вводу вікна.
#
# ТУТ ІНТЕРФЕЙС ЗАЛИШАЄМО — підстановка реальна: для тестів без екрана зручний
# HeadlessRenderer(Renderer), що нічого не малює (я вже користувався таким
# фейком, коли перевіряв App без камери). Друга реалізація = інтерфейс окупився.

from abc import ABC, abstractmethod

import cv2


class Renderer(ABC):
    """Контракт візуалізації."""

    @abstractmethod
    def set_mouse_callback(self, callback):
        """Підписати функцію на події миші у вікні."""

    @abstractmethod
    def draw(self, frame, detections, target_id, fps, target_lost):
        """Намалювати кадр з усіма позначками й показати його."""

    @abstractmethod
    def is_open(self) -> bool:
        """Чи вікно ще відкрите."""


class CvRenderer(Renderer):
    """Реалізація на OpenCV (вікно на екрані)."""

    def __init__(self, window_name="avis"):
        self._window = window_name
        cv2.namedWindow(self._window)

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self._window, callback)

    def draw(self, frame, detections, target_id, fps, target_lost):
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            is_target = det.id == target_id
            color = (0, 0, 255) if is_target else (0, 255, 0)   # ціль — червона (BGR)
            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"id={det.id}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if target_lost:
            cv2.putText(frame, "TARGET LOST", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(self._window, frame)

    def is_open(self) -> bool:
        return cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE) >= 1
