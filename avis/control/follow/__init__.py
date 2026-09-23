# Пакет стеження: координатор + незалежні осі.
# Назовні віддаємо лише те, що потрібно решті системи.

from avis.control.follow.config import FollowConfig
from avis.control.follow.controller import FollowController

__all__ = ["FollowController", "FollowConfig"]
