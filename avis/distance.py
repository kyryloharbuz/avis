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


class DistanceEstimator:
    """Переводить висоту рамки в метри. Потребує одноразового калібрування."""

    def __init__(self, path=CONFIG):
        self._path = path
        self._k = None          # константа калібрування (метри × пікселі)
        self._ref_height = None  # висота рамки на еталонній дистанції
        self._ref_distance = None
        self.load()

    # ── Калібрування ──────────────────────────────────────────────────
    def calibrate(self, distance_m, box_height_px):
        """Запам'ятати: на відстані distance_m рамка мала висоту box_height_px."""
        if box_height_px <= 0:
            raise ValueError("висота рамки має бути додатною")
        self._k = distance_m * box_height_px
        self._ref_distance = distance_m
        self._ref_height = box_height_px
        self.save()
        return self._k

    @property
    def calibrated(self) -> bool:
        return self._k is not None

    # ── Використання ──────────────────────────────────────────────────
    def meters(self, box_height_px):
        """Відстань у метрах, або None якщо ще не калібровано."""
        if self._k is None or box_height_px <= 0:
            return None
        return self._k / box_height_px

    def height_for(self, distance_m):
        """Зворотна задача: яку висоту рамки очікувати на заданій відстані.
        Потрібно, щоб задавати БАЖАНУ дистанцію в метрах, а не в пікселях."""
        if self._k is None or distance_m <= 0:
            return None
        return self._k / distance_m

    # ── Збереження на диск ────────────────────────────────────────────
    def save(self):
        with open(self._path, "w") as f:
            json.dump({"k": self._k,
                       "ref_distance_m": self._ref_distance,
                       "ref_height_px": self._ref_height}, f, indent=2)

    def load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._k = data.get("k")
            self._ref_distance = data.get("ref_distance_m")
            self._ref_height = data.get("ref_height_px")
        except Exception:
            pass          # биті калібрування ігноруємо — просто лишимось без нього

    def describe(self):
        if not self.calibrated:
            return "дистанція НЕ калібрована (запусти calibrate.py)"
        return (f"калібровано: {self._ref_height:.0f}px = {self._ref_distance:.2f} м "
                f"(K={self._k:.0f})")
