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
    def draw(self, frame, detections, target_id, fps, status, predicted_center=None):
        """Намалювати кадр з позначками й показати його.
        status — "VIS" (надійно видно) / "PRED" (ведемо за прогнозом) / "LOST".
        predicted_center — (x, y) поточної гіпотези Калмана, якщо є."""

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

    def draw(self, frame, detections, target_id, fps, status, predicted_center=None):
        # status може бути складеним: "VIS|AUTO" (трекінг|стан польоту).
        # Для ЛОГІКИ (колір, приціл) беремо лише перший складник, а показуємо
        # рядок цілком — щоб було видно і трекінг, і режим польоту.
        base = status.split("|")[0]
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            is_target = det.id == target_id
            color = (0, 0, 255) if is_target else (0, 255, 0)   # ціль — червона (BGR)
            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Підпис: назва класу + track id, напр. "chair #105".
            # Якщо назви немає (напр. у тестових заглушках) — показуємо лише id.
            label = f"{det.name} #{det.id}" if det.name else f"id={det.id}"
            # 0.4 — дрібний шрифт; LINE_AA згладжує, щоб дрібний текст читався.
            cv2.putText(frame, label, (x1, max(11, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ГІПОТЕЗА КАЛМАНА. Раніше при оклюзії на екрані не було НІЧОГО — здавалось,
        # що трекінг помер, хоча він працював. Тепер малюємо, де система "вважає"
        # ціль: жовтий приціл у режимі PRED (ведемо наосліп).
        if predicted_center is not None and base == "PRED":
            px, py = map(int, predicted_center)
            cv2.circle(frame, (px, py), 18, (0, 255, 255), 2)          # жовте коло
            cv2.line(frame, (px - 26, py), (px + 26, py), (0, 255, 255), 1)
            cv2.line(frame, (px, py - 26), (px, py + 26), (0, 255, 255), 1)
            cv2.putText(frame, "predicted", (px + 22, py - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Статус-рядок: одразу видно режим і за яким id ведемо ціль. Це ще й
        # спосіб помітити хибне перезахоплення (id раптом змінився на чужий).
        status_color = {"VIS": (0, 255, 0), "PRED": (0, 255, 255)}.get(base, (0, 0, 255))
        if target_id is None:
            status_label = status
        else:
            # Дістаємо назву цілі з поточних детекцій (якщо вона зараз видима).
            target_name = next((d.name for d in detections if d.id == target_id), "")
            status_label = f"{status}  {target_name} #{target_id}".rstrip()
        cv2.putText(frame, status_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow(self._window, frame)

    def is_open(self) -> bool:
        return cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE) >= 1
