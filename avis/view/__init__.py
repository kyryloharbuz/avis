# Пакет "view" (відображення): показ картинки оператору й ввід із вікна.
# Ре-експортуємо і контракт (Renderer), і реалізацію (CvRenderer).

from avis.view.renderer import CvRenderer, HeadlessRenderer, Renderer

__all__ = ["Renderer", "CvRenderer", "HeadlessRenderer"]
