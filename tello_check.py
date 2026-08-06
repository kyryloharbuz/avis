# Діагностика Tello ПЕРЕД тим, як чіпати основну програму.
# Дрон НЕ злітає — це лише перевірка зв'язку й відео.
#
# Навіщо окремо: політ додає фізику, а фізику й код не варто відлагоджувати
# одночасно. Спершу переконайся, що бачиш і детектуєш — потім замикай контур.
#
# Запуск:
#   python tello_check.py           — крок 2: тільки зв'язок (батарея, температура)
#   python tello_check.py --video   — крок 3: + вікно з відео (q або Esc = вихід)

import sys
import time


def check_connection(tello):
    """Крок 2: чи відповідає дрон узагалі."""
    print("Підключаюсь до 192.168.10.1:8889 ...")
    tello.connect()                      # кине помилку, якщо відповіді немає

    battery = tello.get_battery()
    temperature = tello.get_temperature()
    print(f"  battery:     {battery}%")
    print(f"  temperature: {temperature}°C")
    print(f"  height:      {tello.get_height()} cm")
    print("Зв'язок є.")

    # Команду 'sdk?' розуміє лише SDK 2.0 (Tello EDU). На звичайному Tello вона
    # мовчить і скрипт марно чекає 7 секунд — тому запитуємо ТІЛЬКИ на вимогу.
    if "--sdk" in sys.argv:
        print(f"  SDK: {tello.query_sdk_version()}")

    # --- ПЕРЕВІРКИ БЕЗПЕКИ (не косметика — від них залежить, чи полетить) ---
    ok = True
    if battery < 20:
        print(f"\n  СТОП: заряд {battery}% — Tello відмовиться злітати (<20%).")
        print("        Заряджай до 50%+ перед тестом відео й до 70%+ перед польотом.")
        ok = False
    elif battery < 50:
        print(f"\n  Увага: заряду {battery}% вистачить ненадовго (політ ~13 хв на 100%).")

    # Tello охолоджується потоком від пропелерів. Стоячи ввімкненим на столі
    # він швидко перегрівається і йде в автовимкнення близько 90°C.
    if temperature >= 85:
        print(f"\n  СТОП: {temperature}°C — перегрів, дрон скоро вимкнеться сам.")
        print("        Вимкни його, дай охолонути ~10 хв. Не тримай увімкненим на столі.")
        ok = False

    return ok


def check_video(tello):
    """Крок 3: чи йде живе відео (візьми дрон у руку й поводи)."""
    import cv2
    from djitellopy import Tello

    # --hq = лишити 720p як є. Без прапорця вмикаємо режим низької затримки:
    # 480p достатньо, бо детектор усе одно стискає кадр до 320px.
    if "--hq" not in sys.argv:
        for label, setter, value in (
            ("480p", tello.set_video_resolution, Tello.RESOLUTION_480P),
            ("30fps", tello.set_video_fps, Tello.FPS_30),
            ("3Mbps", tello.set_video_bitrate, Tello.BITRATE_3MBPS),
        ):
            try:
                setter(value)
                print(f"  відео: {label} ok")
            except Exception as e:
                print(f"  відео: {label} не підтримується ({type(e).__name__})")

    tello.streamon()
    reader = tello.get_frame_read()
    print("Чекаю на перший кадр ...")

    # ВАЖЛИВО: до приходу відео djitellopy тримає ЧОРНУ ЗАГЛУШКУ (не None),
    # тому перевіряємо саме .any() — чи є ненульові пікселі.
    deadline = time.time() + 10
    while time.time() < deadline and not (reader.frame is not None and reader.frame.any()):
        time.sleep(0.05)

    if not (reader.frame is not None and reader.frame.any()):
        print("Відео не пішло за 10 с. Див. підказки нижче.")
        tello.streamoff()
        return

    h, w = reader.frame.shape[:2]
    print(f"Відео пішло: {w}x{h}. Вікно відкрито — q або Esc для виходу.")

    frames = 0
    t0 = time.time()
    try:
        while True:
            frame = reader.frame
            if frame is None or not frame.any():
                continue
            # Tello віддає RGB (PIL), OpenCV чекає BGR — без цього все синє.
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames += 1
            fps = frames / max(time.time() - t0, 1e-6)
            cv2.putText(bgr, f"{fps:.1f} FPS  {w}x{h}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("tello check", bgr)
            key = cv2.waitKey(1)
            if key != -1 and (key & 0xFF) in (ord('q'), 27):
                break
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        print(f"Середній FPS: {frames / max(time.time() - t0, 1e-6):.1f}")


def main():
    from djitellopy import Tello

    tello = Tello()
    try:
        healthy = check_connection(tello)
        if "--video" in sys.argv:
            # Не женемо відео на розрядженому/перегрітому дроні: він вимкнеться
            # посеред тесту, і ти шукатимеш баг там, де його немає.
            if healthy or "--force" in sys.argv:
                check_video(tello)
            else:
                print("\nТест відео пропущено (див. СТОП вище).")
                print("Коли усунеш — запусти: python tello_check.py --video")
                print("Або примусово, на свій ризик:  ... --video --force")
    except Exception as e:
        # Друкуємо не лише помилку, а й ЩО з нею робити.
        print(f"\nПОМИЛКА: {type(e).__name__}: {e}\n")
        print("Найчастіші причини:")
        print("  1. Не та мережа — маєш бути під'єднаний до Wi-Fi 'TELLO-XXXXXX',")
        print("     а не до домашньої. Підключення до дрона ЗАБИРАЄ інтернет.")
        print("  2. Увімкнений VPN — вимкни.")
        print("  3. Фаєрвол блокує UDP — System Settings → Network → Firewall.")
        print("  4. Завис попередній запуск — закрий старі процеси python.")
        print("  5. 'Unknown command' на streamon — онови прошивку через застосунок Tello.")
        sys.exit(1)
    finally:
        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
