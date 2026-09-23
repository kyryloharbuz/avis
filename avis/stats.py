# СТАТИСТИКА ПРОГОНУ — накопичення й звіт.
#
# Виділено з App із тієї ж причини, що й конвеєр: App має оркеструвати, а не
# рахувати. Раніше тут були словник на 10 ключів, метод накопичення й метод
# друку — три різні речі всередині класу, який мав керувати циклом.
#
# Класи всередині:
#   RunStats — стани, помилка, САТУРАЦІЯ і ДРОЖ (метрики тюнінгу)
#   FollowScore живе окремо в metrics.py — це метрика ЯКОСТІ за Andon Labs,
#   у неї інша роль: не "як налаштовано", а "наскільки добре стежимо".

from avis.pipeline import LOST, PRED, VIS


class RunStats:
    """Копить показники прогону для ЧЕСНОГО порівняння тюнінгів."""

    def __init__(self):
        self.frames = 0
        self.by_status = {VIS: 0, PRED: 0, LOST: 0}
        self._sum_ex = 0.0
        self._sum_ey = 0.0
        self._tracked = 0
        # Сатурація: скільки разів supervisor обрізав команду. Знати межі не
        # треба — сам факт обрізання і є сатурація.
        self._sat = {"forward": 0, "yaw": 0, "vertical": 0}
        self._cmd_frames = 0
        # Дрож: наскільки команда скакнула з минулого кадру. Велика дрож —
        # дрон смикається (завеликий kp або надто вузька зона спокою).
        self._jitter = 0.0
        self._jitter_n = 0
        self._prev_sent = None
        # Побиті кадри (втрата UDP-пакетів). Це показник ЯКОСТІ КАНАЛУ, а не
        # налаштувань: якщо він високий, жоден коефіцієнт не допоможе — треба
        # прибирати перешкоди в 2.4 ГГц або летіти ближче до ноутбука.
        self._corrupt = 0

    def add(self, result, sent, frame_w, frame_h):
        self.frames += 1
        self.by_status[result.status] = self.by_status.get(result.status, 0) + 1
        if getattr(result, "frame_corrupt", False):
            self._corrupt += 1

        if result.center is not None:
            self._tracked += 1
            self._sum_ex += abs(result.center[0] - frame_w / 2)
            self._sum_ey += abs(result.center[1] - frame_h / 2)

        cmd = result.command
        if cmd is None or sent is None:
            return
        self._cmd_frames += 1
        eps = 0.5
        for axis in self._sat:
            if abs(getattr(sent, axis)) + eps < abs(getattr(cmd, axis)):
                self._sat[axis] += 1
        if self._prev_sent is not None:
            self._jitter += sum(abs(getattr(sent, a) - getattr(self._prev_sent, a))
                                for a in ("yaw", "vertical", "forward"))
            self._jitter_n += 1
        self._prev_sent = sent

    def report(self):
        """Рядки звіту. Саме ці числа порівнюй між прогонами реплею."""
        if self.frames == 0:
            return []
        n = self.frames
        tracked = max(self._tracked, 1)
        cf = self._cmd_frames
        lines = [
            f"ПІДСУМОК: {n} кадрів   "
            f"VIS={self.by_status[VIS]} PRED={self.by_status[PRED]} LOST={self.by_status[LOST]}",
            f"  утримання цілі : {100 * (self.by_status[VIS] + self.by_status[PRED]) / n:.0f}%",
            f"  сер. помилка   : ex={self._sum_ex / tracked:.0f}px  "
            f"ey={self._sum_ey / tracked:.0f}px",
        ]
        if self._corrupt:
            pct = 100 * self._corrupt / n
            note = ("— канал добрий" if pct < 3 else
                    "— помітні перешкоди" if pct < 10 else
                    "— КАНАЛ ПОГАНИЙ, це головна причина втрат цілі")
            lines.append(f"  побиті кадри   : {pct:.0f}%  {note}")
        if cf:
            lines.append(
                f"  сатурація      : вперед {100 * self._sat['forward'] / cf:.0f}%  "
                f"поворот {100 * self._sat['yaw'] / cf:.0f}%  "
                f"вертикаль {100 * self._sat['vertical'] / cf:.0f}%")
            if self._jitter_n:
                lines.append(f"  дрож команд    : {self._jitter / self._jitter_n:.1f} од/кадр")
        return lines
