# КАЛІБРУВАННЯ ДИСТАНЦІЇ — робиться ОДИН раз на камеру.
#
# Ідея (та сама, що ти запропонував, але точніша):
#   стаєш на ВІДОМІЙ відстані від камери → програма міряє висоту рамки в
#   пікселях → рахує константу K = відстань × висота.
#   Далі в польоті: відстань = K / висота_рамки.
#
# Чому висота, а не площа: замір на твоїх польотах показав, що висота рамки
# удвічі стабільніша (2.2% проти 4.4%), бо не залежить від пози й повороту тіла.
#
# Запуск (на дроні або на вебкамері):
#   .venv/bin/python calibrate.py --tello --distance 2.0
#   .venv/bin/python calibrate.py --distance 2.0
#
# Порядок: стань рівно на вказану відстань, повернись обличчям до камери,
# стій рівно і натисни ПРОБІЛ. Програма усереднить кілька кадрів і збереже.

import sys
import time

import cv2

from avis.distance import DistanceEstimator
from avis.perception import Detector


def main():
    distance_m = 2.0
    if "--distance" in sys.argv:
        distance_m = float(sys.argv[sys.argv.index("--distance") + 1])

    if "--tello" in sys.argv:
        from avis.perception.tello_source import TelloSource
        source = TelloSource()
    else:
        from avis.perception import WebcamSource
        source = WebcamSource()

    detector = Detector(classes=[0])        # лише люди
    estimator = DistanceEstimator()
    samples = []

    print("\n" + "=" * 60)
    print(f"КАЛІБРУВАННЯ ДИСТАНЦІЇ на {distance_m:.2f} м")
    print("  1. Стань рівно на цю відстань від камери")
    print("  2. Повернись обличчям, стій прямо, повністю в кадрі")
    print("  3. Натисни ПРОБІЛ (зберемо 15 кадрів і усередним)")
    print("  q / Esc — вийти")
    print("=" * 60 + "\n")
    print(estimator.describe())

    cv2.namedWindow("calibrate")
    collecting = False
    try:
        while True:
            frame = source.read()
            if frame is None:
                continue
            dets = detector.detect(frame)
            # Беремо НАЙБІЛЬШУ людину в кадрі — це той, хто найближче, тобто ти.
            person = max(dets, key=lambda d: d.area) if dets else None

            if person is not None:
                x1, y1, x2, y2 = map(int, person.xyxy)
                h_px = y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"висота: {h_px}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if collecting:
                    samples.append(h_px)
                    cv2.putText(frame, f"замір {len(samples)}/15", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if len(samples) >= 15:
                        avg = sum(samples) / len(samples)
                        k = estimator.calibrate(distance_m, avg)
                        print(f"\nГОТОВО: середня висота {avg:.0f}px на {distance_m:.2f} м")
                        print(f"  K = {k:.0f}  (збережено у distance_calibration.json)")
                        print("\nПеревірка — очікувані висоти рамки:")
                        for d in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
                            print(f"    {d:.1f} м → {estimator.height_for(d):.0f}px")
                        break
            else:
                cv2.putText(frame, "людину не видно", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("calibrate", frame)
            key = cv2.waitKey(1)
            codes = {key, key & 0xFF}
            if codes & {27, ord('q')}:
                print("Скасовано.")
                break
            if ord(' ') in codes and person is not None and not collecting:
                collecting = True
                samples = []
                print("Збираю заміри — стій рівно ...")
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
