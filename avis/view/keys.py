# Reading individual keys FROM THE TERMINAL, without Enter.
#
# WHY A SEPARATE MODULE: on macOS an OpenCV window launched from a terminal
# often does not receive keyboard focus -- the keystrokes go to the shell
# ("zsh: command not found: t") and the program never sees them. This problem
# turns up anywhere there is interaction, so the fix lives in one place.
#
# The mechanics: the terminal is switched to cbreak mode (it yields a character
# immediately, without waiting for Enter) and reads are non-blocking via
# select() -- so the video does not stall.

import select
import sys
import termios
import tty


class TerminalKeys:
    def __init__(self):
        self._fd = None
        self._old = None
        # In an IDE run window stdin may not be a terminal -- in that case we
        # disable ourselves quietly and control is left to the OpenCV window.
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    @property
    def active(self) -> bool:
        return self._fd is not None

    def get(self):
        """The key pressed, or None if nothing was pressed."""
        if self._fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        """MUST be called in a finally block -- otherwise the terminal is left
        broken (you will not see what you type). The `reset` command rescues it."""
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._fd = None
