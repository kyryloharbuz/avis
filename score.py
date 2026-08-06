# Оцінка ЯКОСТІ стеження на записаних польотах — метрика у стилі
# Andon Labs Drone-Bench (задача "Follow", людський базлайн 67.4%).
#
# Читає ТІЛЬКИ .jsonl (лог), відео не потрібне. Це навмисно:
#   • швидко — 8000 кадрів за частку секунди, без декодування H.264;
#   • надійно — якщо політ обірвався примусово, .mp4 лишається недописаним і
#     нечитабельним, а .jsonl дописується щокадру й переживає це.
#
# Запуск:
#   .venv/bin/python score.py                    — усі записи
#   .venv/bin/python score.py flight_2026...     — один запис
#   .venv/bin/python score.py --area 90000       — інша бажана дистанція

import glob
import json
import os
import sys

from avis.control.controller import FollowController
from avis.metrics import HUMAN_BASELINE, FollowScore

# Кадр Tello. Використовується, якщо в записі немає розмірів (старі файли).
DEFAULT_W, DEFAULT_H = 960, 720


def score_file(path, desired_area, auto_only=False):
    """Порахувати Follow-скор для одного запису.
    auto_only=True — рахувати ЛИШЕ кадри, де дрон реально стежив (стан AUTO).
    Це чесніше як "епізод стеження": кадри до вибору цілі й на землі не є
    невдачею стеження, їх туди зараховувати нема за що."""
    fs = FollowScore(desired_area=desired_area)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if auto_only and r.get("state") != "AUTO":
                continue
            w = r.get("frame_w") or DEFAULT_W
            h = r.get("frame_h") or DEFAULT_H
            tid = r.get("target_id")
            # Шукаємо рамку цілі серед детекцій цього кадру.
            target = None
            if tid is not None:
                target = next((d for d in r["detections"] if d["id"] == tid), None)
            if target is None:
                fs.add(None, 0.0, w, h)        # цілі не вели — нуль балів
                continue
            x1, y1, x2, y2 = target["xyxy"]
            fs.add(((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1) * (y2 - y1), w, h)
    return fs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # За замовчуванням беремо ТУ САМУ уставку, що й у регуляторі, — інакше
    # метрика міряла б дистанцію за іншим еталоном, ніж система нею керує.
    desired_area = FollowController()._desired_area
    if "--area" in sys.argv:
        desired_area = float(sys.argv[sys.argv.index("--area") + 1])

    files = ([os.path.join("recordings", a + ".jsonl") for a in args] if args
             else sorted(glob.glob("recordings/*.jsonl")))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("Записів не знайдено в recordings/")
        return

    print(f"\nFOLLOW-СКОР — ціль у центральній третині кадру І на потрібній дистанції")
    print(f"Людський базлайн Andon Labs Drone-Bench: {100 * HUMAN_BASELINE['Follow']:.1f}%")
    print(f"Бажана площа рамки (дистанція): {desired_area:.0f}\n")
    print(f"{'запис':<22} {'кадрів':>7} {'ВЕСЬ ПРОГІН':>12} {'ПІД ЧАС СТЕЖЕННЯ':>18}")
    print("-" * 64)

    tot_g = tot_n = auto_g = auto_n = 0
    for path in files:
        name = os.path.basename(path)[:-6]
        full = score_file(path, desired_area)
        auto = score_file(path, desired_area, auto_only=True)
        a_txt = f"{100 * auto.score:.1f}%" if auto.frames else "—"
        print(f"{name.replace('flight_', ''):<22} {full.frames:>7} "
              f"{100 * full.score:>11.1f}% {a_txt:>18}")
        tot_g += full.good
        tot_n += full.frames
        auto_g += auto.good
        auto_n += auto.frames

    print("-" * 64)
    print(f"{'РАЗОМ':<22} {tot_n:>7} {100 * tot_g / max(tot_n, 1):>11.1f}% "
          f"{100 * auto_g / max(auto_n, 1):>17.1f}%")
    print("\n  ВЕСЬ ПРОГІН      — включно з часом до вибору цілі й на землі (песимістично)")
    print("  ПІД ЧАС СТЕЖЕННЯ — лише кадри в режимі AUTO (це і є епізод стеження)")
    print("\n  Чесні відмінності від оригіналу: ми не детектуємо ЗІТКНЕННЯ (в них воно")
    print("  обнуляє решту кадрів), а дистанцію міряємо через площу рамки, не напряму.")


if __name__ == "__main__":
    main()
