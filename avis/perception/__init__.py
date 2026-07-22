# Пакет "perception" (сприйняття): усе, що перетворює сирі кадри на розуміння
# сцени — джерело кадрів → детекція → вибір цілі → згладжування позиції.
#
# Пам'ятаєш питання "чому __init__.py порожній"? Ось приклад КОРИСНОГО
# наповнення. Ми РЕ-ЕКСПОРТУЄМО публічні класи підпакета, щоб зовні писати
# коротко:  from avis.perception import Detector
# замість довгого:  from avis.perception.detector import Detector
# Це "фасад" пакета — його публічне обличчя.
#
# НАГАДУВАННЯ (не як у Java): цей пакет НЕ захищає дані. Немає package-private —
# будь-хто все одно може зробити `from avis.perception.detector import Detector`.
# Пакет = неймспейс + структура, а не контроль доступу.

from avis.perception.detector import Detector
from avis.perception.frame_source import FrameSource, WebcamSource
from avis.perception.target_selector import TargetSelector
from avis.perception.tracking_filter import KalmanFilter

# __all__ — офіційний "публічний список" пакета: що саме експортується при
# `from avis.perception import *` і що редактор показує як публічний API.
__all__ = ["FrameSource", "WebcamSource", "Detector", "TargetSelector", "KalmanFilter"]
