# ОЦІНКА ДИСТАНЦІЇ ДО ЦІЛІ В МЕТРАХ.
#
# Досі дистанцію ми оцінювали через ПЛОЩУ рамки — але це погана міра:
# площа залежить і від ширини, а ширина стрибає від пози (руки в боки,
# поворот тіла). Замір на 1295 вибірках із реальних польотів:
#     площа  — коефіцієнт варіації 4.4%
#     висота — 2.2%   ← удвічі стабільніша
#
# ФІЗИКА (модель камери-обскури): для об'єкта сталої реальної висоти
#     висота_в_пікселях = f * реальна_висота / відстань
# тобто відстань обернено пропорційна висоті рамки:
#     ВІДСТАНЬ = K / висота_в_пікселях
# де K — одна константа, яку треба ЗАМІРЯТИ один раз (калібрування):
#     K = відома_відстань * висота_рамки_на_цій_відстані
#
# Калібрування зберігається у файл, тож робиться ОДИН раз на камеру.

import json
import os

CONFIG = "distance_calibration.json"

# НОМІНАЛЬНА константа для НЕкаліброваного випадку. Взята з реального
# калібрування Tello (318px = 2.00 м). Навіщо вона потрібна: без неї метод
# meters() повертав None, і в системі зʼявлялась ДРУГА гілка керування —
# через площу рамки, зі своєю окремою уставкою. Дві уставки однієї величини
# не збігались і мовчки ламали рух за ціллю. Тепер шлях один: метри завжди,
# просто без калібрування вони приблизні.
NOMINAL_K = 635.0

# СПІВВІДНОШЕННЯ зріст/ширина рамки людини. Заміряно на 7988 чистих (не
# обрізаних) вимірах із реальних польотів: медіана 2.26.
# Потрібне, щоб вивести константу для ШИРИНИ зі старого калібрування по висоті.
DEFAULT_ASPECT = 2.26


class DistanceEstimator:
    """Переводить висоту рамки в метри. Потребує одноразового калібрування."""

    def __init__(self, path=CONFIG):
        self._path = path
        self._k = None           # константа для ВИСОТИ (метри × пікселі)
        self._k_width = None     # константа для ШИРИНИ
        self._ref_height = None
        self._ref_width = None
        self._ref_distance = None
        self.load()

    # ── Дві міри замість однієї ───────────────────────────────────────
    #
    # ЧОМУ ДВІ. Висота точніша (стабільність 1.7% проти 2.2%), але людина
    # перестає вміщатись у кадр 720px ближче ніж 0.88 м — і висота "зависає",
    # роблячи систему сліпою саме зблизька. Ширина ж (плечі) лишається в кадрі
    # аж до ~0.29 м. Тому: доки рамка ціла — міряємо ВИСОТОЮ, а щойно її
    # обрізало зверху/знизу — переходимо на ШИРИНУ.
    @property
    def k_width(self):
        """Константа для ширини. Якщо калібрували лише висоту — виводимо її
        через заміряне співвідношення зріст/ширина."""
        if self._k_width is not None:
            return self._k_width
        base = self._k if self._k is not None else NOMINAL_K
        return base / DEFAULT_ASPECT

    @staticmethod
    def frame_fill(xyxy, frame_w, frame_h):
        """Яку ЧАСТКУ кадру займає рамка (0..1) — ПРЯМИЙ сигнал близькості.

        Головна перевага над оцінкою дистанції: не залежить НІ від
        калібрування, НІ від того, яку міру обрано, НІ від пози. Рамка на
        півекрана означає "дуже близько" за будь-яких обставин.
        Саме тому це надійний ЗАПОБІЖНИК, а не заміна вимірюванню."""
        x1, y1, x2, y2 = xyxy
        w = max(0.0, min(x2, frame_w) - max(x1, 0.0))
        h = max(0.0, min(y2, frame_h) - max(y1, 0.0))
        return (w * h) / float(frame_w * frame_h)

    def meters_from_box(self, xyxy, frame_w, frame_h, margin=2):
        """Відстань за рамкою, з автоматичним вибором міри.
        Повертає (відстань_м, якою_мірою) або (None, None)."""
        x1, y1, x2, y2 = xyxy
        height, width = y2 - y1, x2 - x1
        clipped_v = y1 <= margin or y2 >= frame_h - margin
        clipped_h = x1 <= margin or x2 >= frame_w - margin

        if not clipped_v and height > 0:
            return self.meters(height), "height"
        if not clipped_h and width > 0:
            return self.k_width / width, "width"
        return None, None

    # ── Калібрування ──────────────────────────────────────────────────
    def calibrate(self, distance_m, box_height_px, box_width_px=None):
        """Запам'ятати розміри рамки на ВІДОМІЙ відстані.
        Ширину варто передавати — саме вона працює на близьких дистанціях."""
        if box_height_px <= 0:
            raise ValueError("висота рамки має бути додатною")
        self._k = distance_m * box_height_px
        self._ref_distance = distance_m
        self._ref_height = box_height_px
        if box_width_px and box_width_px > 0:
            self._k_width = distance_m * box_width_px
            self._ref_width = box_width_px
        self.save()
        return self._k

    @property
    def calibrated(self) -> bool:
        return self._k is not None

    # ── Використання ──────────────────────────────────────────────────
    def meters(self, box_height_px):
        """Відстань у метрах. ЗАВЖДИ повертає число (окрім некоректної рамки):
        без калібрування використовує номінальну константу. Це навмисно —
        щоб у системі був ОДИН шлях керування дистанцією, а не два."""
        if box_height_px <= 0:
            return None
        return (self._k if self._k is not None else NOMINAL_K) / box_height_px

    def height_for(self, distance_m):
        """Зворотна задача: яку висоту рамки очікувати на заданій відстані."""
        if distance_m <= 0:
            return None
        return (self._k if self._k is not None else NOMINAL_K) / distance_m

    # ── Збереження на диск ────────────────────────────────────────────
    def save(self):
        with open(self._path, "w") as f:
            json.dump({"k": self._k,
                       "k_width": self._k_width,
                       "ref_distance_m": self._ref_distance,
                       "ref_height_px": self._ref_height,
                       "ref_width_px": self._ref_width}, f, indent=2)

    def load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._k = data.get("k")
            self._k_width = data.get("k_width")
            self._ref_distance = data.get("ref_distance_m")
            self._ref_height = data.get("ref_height_px")
            self._ref_width = data.get("ref_width_px")
        except Exception:
            pass          # биті калібрування ігноруємо — просто лишимось без нього

    def describe(self):
        if not self.calibrated:
            return ("дистанція НЕ калібрована — працюю за номінальною K=%.0f, "
                "точність приблизна (запусти calibrate.py)" % NOMINAL_K)
        w = (f", ширина K={self._k_width:.0f}" if self._k_width
             else f", ширина K={self.k_width:.0f} (виведено)")
        return (f"калібровано: {self._ref_height:.0f}px = {self._ref_distance:.2f} м "
                f"(K={self._k:.0f}{w})")
