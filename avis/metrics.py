# МЕТРИКИ ЯКОСТІ СТЕЖЕННЯ у стилі Andon Labs Drone-Bench.
#
# Навіщо: досі ми міряли FPS — це метрика ШВИДКОСТІ, а не ЯКОСТІ. FPS не каже
# нічого про те, чи дрон справді тримає ціль. Andon Labs у своєму бенчмарку
# (задача "Follow") міряють саме якість, і в них є людський базлайн — 67.4%.
# Маючи ту саму метрику, ми можемо чесно сказати "моя система дає X%",
# а не "начебто працює".
#
# ВИЗНАЧЕННЯ (як у них, з нашими уточненнями):
#   кадр зараховується, якщо ціль
#     а) у ЦЕНТРАЛЬНІЙ ТРЕТИНІ кадру по обох осях, І
#     б) у прийнятному діапазоні ДИСТАНЦІЇ.
#   Підсумок = частка зарахованих кадрів по всьому прогону.
#
# ЧЕСНІ ВІДМІННОСТІ від оригіналу (щоб не видавати бажане за дійсне):
#   • вони обнуляють решту кадрів після ЗІТКНЕННЯ — ми зіткнення не детектуємо,
#     тож наша оцінка ОПТИМІСТИЧНІША за їхню;
#   • дистанцію вони міряють реальну, ми — через площу рамки (проксі);
#   • їхній прогін — повний сценарій в офісі, наш — довільний запис.
# Тому число порівнюване лише ОРІЄНТОВНО, і про це треба писати прямо.

# Людські базлайни Andon Labs Drone-Bench (для контексту у звіті).
HUMAN_BASELINE = {
    "Reconstruct": 0.822,
    "Localize": 0.840,
    "Navigate": 0.920,
    "Detect": 0.721,
    "Follow": 0.674,
}


def in_center_third(cx, cy, frame_w, frame_h):
    """Чи центр цілі в центральній третині кадру по ОБОХ осях.
    Центральна третина по ширині — це [W/3, 2W/3], тобто |cx - W/2| < W/6."""
    return abs(cx - frame_w / 2) < frame_w / 6 and abs(cy - frame_h / 2) < frame_h / 6


def in_distance_band(area, desired_area, near=2.5, far=0.4):
    """Чи ціль на прийнятній дистанції. Дистанцію оцінюємо через площу рамки:
    площа ~ 1/відстань². Смуга [far*desired, near*desired] відповідає приблизно
    0.63x..1.6x бажаної відстані — тобто "не впритул і не загубився вдалині"."""
    if not desired_area:
        return True
    return far * desired_area <= area <= near * desired_area


class FollowScore:
    """Накопичує Follow-метрику по кадрах прогону."""

    def __init__(self, desired_area=60000.0, estimator=None):
        self._desired_area = desired_area
        # Необов'язковий оцінювач дистанції: якщо камеру відкалібровано,
        # у звіті з'явиться реальна відстань у МЕТРАХ, а не абстрактна площа.
        self._estimator = estimator
        self._distances = []
        self.frames = 0          # усі кадри прогону
        self.centered = 0        # ціль у центральній третині
        self.in_range = 0        # ціль на прийнятній дистанції
        self.good = 0            # і те, і те одночасно (це і є Follow-скор)
        self.no_target = 0       # кадри, де цілі не було взагалі

    def add(self, center, area, frame_w, frame_h, box_height=None):
        """center=None означає, що ціль у цьому кадрі не вели (нуль балів).
        box_height — висота рамки в пікселях, для оцінки дистанції в метрах."""
        self.frames += 1
        if center is None:
            self.no_target += 1
            return
        if self._estimator is not None and box_height:
            m = self._estimator.meters(box_height)
            if m is not None:
                self._distances.append(m)
        c = in_center_third(center[0], center[1], frame_w, frame_h)
        d = in_distance_band(area, self._desired_area)
        self.centered += c
        self.in_range += d
        self.good += (c and d)

    @property
    def score(self):
        """Головне число: частка кадрів, де ціль була в цільовій зоні."""
        return self.good / self.frames if self.frames else 0.0

    def report(self):
        """Рядки звіту — і для консолі, і для README."""
        n = self.frames or 1
        return [
            f"FOLLOW-СКОР: {100 * self.score:.1f}%   "
            f"(людський базлайн Andon Labs — {100 * HUMAN_BASELINE['Follow']:.1f}%)",
            f"  у центральній третині : {100 * self.centered / n:.1f}%",
            f"  на потрібній дистанції: {100 * self.in_range / n:.1f}%",
            f"  цілі не було взагалі  : {100 * self.no_target / n:.1f}%",
        ] + self._distance_lines()

    def _distance_lines(self):
        """Реальна дистанція в метрах — лише якщо камеру відкалібровано."""
        if not self._distances:
            return []
        d = sorted(self._distances)
        return [
            f"  дистанція до цілі     : медіана {d[len(d) // 2]:.2f} м  "
            f"(від {d[0]:.2f} до {d[-1]:.2f})",
        ]
