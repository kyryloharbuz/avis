# РУЧНЕ керування Tello з клавіатури + відео. БЕЗ автономності й БЕЗ PID.
#
# Це навмисно проміжний крок (крок 3 roadmap): спершу переконайся, що дрон
# слухає команди й ти вмієш його посадити, і лише потім замикай контур на AI.
# Відлагоджувати фізику й алгоритм одночасно — найкоротший шлях зламати дрон.
#
# Запуск:  .venv/bin/python tello_fly.py
#
# ┌─ КЕРУВАННЯ ─────────────────────────────────────────────────┐
# │  T — злетіти            L — сісти (основний спосіб)         │
# │  W / S — вперед / назад     A / D — вліво / вправо          │
# │  Q / E — поворот ліворуч / праворуч                         │
# │  R / F — вгору / вниз                                       │
# │  X — АВАРІЙНЕ вимкнення моторів (дрон ПАДАЄ! лише як край)  │
# │  Esc — сісти й вийти                                        │
# └─────────────────────────────────────────────────────────────┘
#
# ПРАВИЛА БЕЗПЕКИ (прочитай перед першим злетом):
#   1. Приміщення: вільні 3x3 м, стеля від 2.5 м, ніяких людей поруч.
#   2. Захист пропелерів — надягнути.
#   3. Прибрати крихке, закрити вікна, вимкнути протяг.
#   4. Заряд > 50%. Килим/підлога рівні (Tello злітає з рівної поверхні).
#   5. Тримай палець на L. Не панікуй — L садить м'яко, X роняє.
#   6. Не літай над людьми й тваринами.

import select
import sys
import termios
import time
import tty

import cv2
from djitellopy import Tello


class TerminalKeys:
    """Читає окремі клавіші прямо з ТЕРМІНАЛУ, без Enter.

    НАВІЩО: на macOS вікно OpenCV, запущене з терміналу, часто не отримує
    фокус клавіатури — натискання йдуть у shell ("zsh: command not found: t"),
    а дрон команд не бачить. Тому не покладаємось на фокус вікна: слухаємо
    клавіші з терміналу, який фокус уже має.

    Механіка: переводимо термінал у режим cbreak (віддає символ одразу, не
    чекаючи Enter) і читаємо неблокуюче через select().
    """

    def __init__(self):
        self._fd = None
        self._old = None
        # У PyCharm Run-вікні stdin може не бути терміналом — тоді просто
        # не вмикаємо цей режим (лишиться керування через вікно OpenCV).
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    @property
    def active(self):
        return self._fd is not None

    def get(self):
        """Повертає натиснутий символ або None, якщо нічого не натиснуто."""
        if self._fd is None:
            return None
        # select із нульовим таймаутом = "перевір і одразу повернись".
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        """ОБОВ'ЯЗКОВО викликати — інакше термінал лишиться зіпсованим."""
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

SPEED = 35          # швидкість горизонталі (-100..100). Помірна = безпечна.
SPEED_VERTICAL = 40  # вгору/вниз
SPEED_YAW = 60       # поворот (його важче відчути, тому трохи більше)


def overlay(frame, tello, flying, last_key):
    """Малює телеметрію поверх кадру — щоб не переводити погляд на консоль."""
    state = "У ПОВІТРІ" if flying else "на землі"
    color = (0, 165, 255) if flying else (0, 255, 0)
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    try:
        info = f"bat {tello.get_battery()}%  h {tello.get_height()}cm"
    except Exception:
        info = "телеметрія недоступна"
    cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "T=злет  L=посадка  WASD/QE/RF  X=аварія", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if last_key:
        cv2.putText(frame, last_key, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    print(f"Заряд: {battery}%")
    if battery < 30:
        print("СТОП: для польоту потрібно щонайменше 30% (краще 50%+). Заряди.")
        sys.exit(1)

    # 480p — менше затримки; для ручного керування це критичніше за якість.
    for setter, value in ((tello.set_video_resolution, Tello.RESOLUTION_480P),
                          (tello.set_video_fps, Tello.FPS_30)):
        try:
            setter(value)
        except Exception:
            pass

    tello.streamon()
    reader = tello.get_frame_read()

    # Чекаємо перший справжній кадр (заглушка — чорна, а не None).
    deadline = time.time() + 10
    while time.time() < deadline and not (reader.frame is not None and reader.frame.any()):
        time.sleep(0.05)

    flying = False
    last_key = ""
    keys = TerminalKeys()          # ввід із терміналу (обхід проблеми фокуса на macOS)

    print("\n" + "=" * 58)
    if keys.active:
        print("Керування працює ПРЯМО ТУТ, у цьому терміналі.")
        print("Не переключайся на вікно з відео — просто натискай клавіші.")
    else:
        print("Термінал недоступний (запуск із IDE) — клікни по вікну з відео.")
    print("  T — злетіти     L — сісти     Esc або Ctrl+C — вихід")
    print("  W/S вперед-назад   A/D вбік   Q/E поворот   R/F вгору-вниз")
    print("=" * 58 + "\n")

    try:
        while True:
            frame = reader.frame
            if frame is None or not frame.any():
                continue
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay(bgr, tello, flying, last_key)
            cv2.imshow("tello manual", bgr)

            # Клавіша з ДВОХ джерел: вікно OpenCV АБО термінал. Що спрацює —
            # те й беремо (waitKey потрібен у будь-якому разі, він оновлює вікно).
            key = cv2.waitKey(1) & 0xFF
            typed = keys.get()
            if typed:
                key = ord(typed.lower()) if typed not in ('\x1b', '\x03') else 27

            # Швидкості цього кадру. ВАЖЛИВО: якщо клавішу не тримають — шлемо
            # нулі, і дрон ВИСИТЬ на місці. Без цього він продовжував би рух
            # останньою командою, бо RC-команда діє до наступної.
            lr = fb = ud = yaw = 0

            if key == ord('t') and not flying:
                print("Злітаю ...")
                tello.takeoff()                     # автозліт на ~80 см
                flying = True
                last_key = "TAKEOFF"
            elif key == ord('l') and flying:
                print("Сідаю ...")
                tello.send_rc_control(0, 0, 0, 0)   # спершу зупинити рух
                tello.land()
                flying = False
                last_key = "LAND"
            elif key == ord('x'):
                # АВАРІЯ: мотори вимикаються миттєво, дрон падає. Тільки якщо
                # він летить у стіну/людину і посадка вже не встигне.
                print("АВАРІЙНА ЗУПИНКА МОТОРІВ")
                tello.emergency()
                flying = False
                last_key = "EMERGENCY"
            elif key == 27:                          # Esc
                break
            elif flying:
                # Рух — лише коли дрон у повітрі.
                if key == ord('w'): fb, last_key = SPEED, "вперед"
                elif key == ord('s'): fb, last_key = -SPEED, "назад"
                elif key == ord('a'): lr, last_key = -SPEED, "вліво"
                elif key == ord('d'): lr, last_key = SPEED, "вправо"
                elif key == ord('r'): ud, last_key = SPEED_VERTICAL, "вгору"
                elif key == ord('f'): ud, last_key = -SPEED_VERTICAL, "вниз"
                elif key == ord('q'): yaw, last_key = -SPEED_YAW, "поворот вліво"
                elif key == ord('e'): yaw, last_key = SPEED_YAW, "поворот вправо"
                elif key == 255: last_key = ""       # нічого не натиснуто

            # RC-команду шлемо КОЖЕН кадр (нулі = висіти). Tello також має
            # власний запобіжник: без команд ~15 с він сідає сам.
            if flying:
                tello.send_rc_control(lr, fb, ud, yaw)

            if cv2.getWindowProperty("tello manual", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nCtrl+C — саджу дрон ...")
    except Exception as e:
        print(f"\nПОМИЛКА: {type(e).__name__}: {e}")
    finally:
        keys.restore()   # повернути нормальний режим терміналу
        # ГАРАНТОВАНА посадка: що б не сталось — помилка, закрите вікно, Ctrl+C —
        # дрон не має лишитись висіти без керування.
        try:
            if flying:
                print("Аварійне завершення — саджу дрон ...")
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
        except Exception:
            pass
        try:
            tello.streamoff()
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Готово.")


if __name__ == "__main__":
    main()
