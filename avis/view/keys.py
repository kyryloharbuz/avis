# Читання окремих клавіш із ТЕРМІНАЛУ, без Enter.
#
# НАВІЩО ОКРЕМИЙ МОДУЛЬ: на macOS вікно OpenCV, запущене з терміналу, часто не
# отримує фокус клавіатури — натискання йдуть у shell ("zsh: command not
# found: t"), а програма їх не бачить. Ця проблема вилазить скрізь, де є
# інтерактив, тому виносимо рішення в одне місце.
#
# Механіка: термінал переводиться в режим cbreak (віддає символ одразу, не
# чекаючи Enter), читання неблокуюче через select() — щоб не гальмувати відео.

import select
import sys
import termios
import tty


class TerminalKeys:
    def __init__(self):
        self._fd = None
        self._old = None
        # У Run-вікні IDE stdin може не бути терміналом — тоді тихо вимикаємось,
        # і лишається керування через вікно OpenCV.
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    @property
    def active(self) -> bool:
        return self._fd is not None

    def get(self):
        """Натиснутий символ або None, якщо нічого не натиснуто."""
        if self._fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        """ОБОВ'ЯЗКОВО викликати у finally — інакше термінал лишиться зіпсованим
        (не буде видно введення). Рятує команда `reset`."""
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._fd = None
