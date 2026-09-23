# Оцінка ЯКОСТІ стеження на записаних польотах — метрика у стилі
# Andon Labs Drone-Bench (задача "Follow", людський базлайн 67.4%).
#
# Читає ТІЛЬКИ .jsonl (лог), відео не потрібне. Це навмисно:
#   • швидко — 8000 кадрів за частку секунди, без декодування H.264;
#   • надійно — якщо політ обірвався примусово, .mp4 лишається недописаним і
#     нечитабельним, а .jsonl дописується щокадру й переживає це.
#
# ЧОМУ ЦЕЙ ФАЙЛ ПЕРЕПИСАНО. Він був побудований навколо бажаної ПЛОЩІ рамки —
# уставки, якої в системі вже не існує (керування перейшло на метри). Через це
# він просто падав з ImportError і мовчки не використовувався. Плюс він
# припускав кадр 960x720, тоді як Tello віддає 648x478, тож перевірка "ціль у
# центральній третині" рахувалась по неіснуючому кадру.
#
# Тепер уставка й спосіб виміру дистанції беруться з ТИХ САМИХ джерел, що й у
# польоті: FollowConfig і DistanceEstimator.meters_from_box.
#
# Запуск:
#   .venv/bin/python score.py                      — усі записи
#   .venv/bin/python score.py flight_2026...       — один запис
#   .venv/bin/python score.py --target 1.2         — інша бажана дистанція, м

import glob
import json
import os
import sys

from avis.control.follow.config import FollowConfig
from avis.distance import DistanceEstimator
from avis.metrics import HUMAN_BASELINE, FollowScore

# Кадр Tello 720p — крайній запасний варіант, якщо розмір не вдалось дізнатись
# ні з логу, ні з відео.
DEFAULT_W, DEFAULT_H = 960, 720


def frame_size(log_path, first_record):
    """Розмір кадру запису. Три джерела, від найнадійнішого до запасного.

    ЧОМУ ЦЕ ВАЖЛИВО, А НЕ ДРІБНИЦЯ. Від розміру залежать ОБИДВА складники
    метрики: де лежить центральна третина і чи торкається рамка краю (а від
    цього — якою мірою міряти дистанцію). Раніше тут стояло просто 960x720,
    тоді як Tello віддає 648x478: центр кадру вважався у точці 480 замість
    324, і скор виходив 0.6% замість справжніх ~23%."""
    w, h = first_record.get("frame_w"), first_record.get("frame_h")
    if w and h:
        return w, h                    # нові записи зберігають розмір самі

    # Старі записи розміру не мають — беремо із ЗАГОЛОВКА відео. Це дешево:
    # контейнер відкривається без декодування жодного кадру.
    video = log_path[:-6] + ".mp4"
    if os.path.exists(video):
        try:
            import cv2
            cap = cv2.VideoCapture(video)
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vw > 0 and vh > 0:
                return vw, vh
        except Exception:
            pass
    return DEFAULT_W, DEFAULT_H


def score_file(path, target_m, tolerance_m, estimator, auto_only=False):
    """Порахувати Follow-скор для одного запису.
    auto_only=True — рахувати ЛИШЕ кадри, де дрон реально стежив (стан AUTO).
    Це чесніше як "епізод стеження": кадри до вибору цілі й на землі не є
    невдачею стеження, їх туди зараховувати нема за що."""
    fs = FollowScore(target_m=target_m, tolerance_m=tolerance_m)
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return fs
    w, h = frame_size(path, records[0])
    for r in records:
        if auto_only and r.get("state") != "AUTO":
            continue
        tid = r.get("target_id")
        # Шукаємо рамку цілі серед детекцій цього кадру.
        target = None
        if tid is not None:
            target = next((d for d in r["detections"] if d["id"] == tid), None)
        if target is None:
            fs.add(None, None, w, h)        # цілі не вели — нуль балів
            continue
        box = tuple(target["xyxy"])
        x1, y1, x2, y2 = box
        # Дистанцію міряємо ТАК САМО, як у польоті: висотою, доки рамка
        # ціла, і шириною плечей, коли зріст уже не вміщається в кадр.
        distance_m, _measure = estimator.meters_from_box(box, w, h)
        fs.add(((x1 + x2) / 2, (y1 + y2) / 2), distance_m, w, h)
    return fs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # За замовчуванням беремо ТУ САМУ уставку, що й у регуляторі, — інакше
    # метрика міряла б дистанцію за іншим еталоном, ніж система нею керує.
    cfg = FollowConfig()
    target_m = cfg.distance_target_m
    if "--target" in sys.argv:
        target_m = float(sys.argv[sys.argv.index("--target") + 1])
    # Смуга допуску — та сама формула, що в App: три зони спокою.
    tolerance_m = 3 * cfg.distance_deadzone_m
    estimator = DistanceEstimator()

    files = ([os.path.join("recordings", a + ".jsonl") for a in args] if args
             else sorted(glob.glob("recordings/*.jsonl")))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("Записів не знайдено в recordings/")
        return

    print("\nFOLLOW-СКОР — ціль у центральній третині кадру І на потрібній дистанції")
    print(f"Людський базлайн Andon Labs Drone-Bench: {100 * HUMAN_BASELINE['Follow']:.1f}%")
    print(f"Бажана дистанція: {target_m:.2f} ± {tolerance_m:.2f} м")
    print(f"{estimator.describe()}\n")
    print(f"{'запис':<22} {'кадрів':>7} {'ВЕСЬ ПРОГІН':>12} {'ПІД ЧАС СТЕЖЕННЯ':>18}")
    print("-" * 64)

    tot_g = tot_n = auto_g = auto_n = 0
    for path in files:
        name = os.path.basename(path)[:-6]
        full = score_file(path, target_m, tolerance_m, estimator)
        auto = score_file(path, target_m, tolerance_m, estimator, auto_only=True)
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
    print("  обнуляє решту кадрів), а дистанцію міряємо за розміром рамки, не напряму.")


if __name__ == "__main__":
    main()
