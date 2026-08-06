# Джерело кадрів із дрона DJI/Ryze Tello — друга реалізація FrameSource.
#
# ЦЕ ТОЙ САМИЙ МОМЕНТ, ЗАРАДИ ЯКОГО РОБИЛИ АБСТРАКЦІЮ. Весь конвеєр (YOLO,
# трекінг, Калман, PID, рендер) лишається БЕЗ ЗМІН — міняється один рядок у
# main.py: source=WebcamSource() → source=TelloSource().
#
# Дві пастки Tello, які закриті всередині цього класу:
#   1) КОЛІР. Tello віддає кадр у RGB (djitellopy робить np.array(frame.to_image()),
#      а PIL — це RGB). OpenCV же всюди очікує BGR. Без конвертації картинка
#      буде синюшна, а YOLO працюватиме по спотвореному кольору.
#   2) "ПОРОЖНІЙ" ПЕРШИЙ КАДР. До приходу відео djitellopy тримає ЧОРНУ
#      ЗАГЛУШКУ np.zeros([300,400,3]) — це НЕ None. Тому наївна перевірка
#      `if frame is None` не спрацює, і в YOLO полетять чорні кадри. Тут ми
#      чекаємо на перший справжній кадр ще в конструкторі.

import time

import cv2
from djitellopy import Tello

from avis.perception.frame_source import FrameSource


class TelloSource(FrameSource):
    """Відеопотік із Tello у форматі BGR (сумісний з рештою конвеєра)."""

    # first_frame_timeout — скільки секунд чекати на перший реальний кадр.
    # low_latency — знизити роздільну здатність до 480p і бітрейт.
    #   ЧОМУ ЦЕ БЕЗКОШТОВНО ДЛЯ НАС: детектор працює з imgsz=320, тобто ВСЕ
    #   ОДНО стискає кадр до 320px. Гнати 720p через Wi-Fi, щоб потім викинути
    #   дві третини пікселів — марна затримка. 480p дає ту саму детекцію, але
    #   менше даних у каналі: нижча затримка і менше артефактів від втрати
    #   пакетів. Вимкни (low_latency=False), якщо потрібна гарна картинка для
    #   демо-відео, а не мінімальний лаг.
    # tello — ГОТОВИЙ об'єкт Tello ззовні. Потрібен, коли той самий дрон має
    #   і віддавати відео, і приймати команди: два незалежні Tello() билися б
    #   за один UDP-порт. Якщо не передати — створимо власний (режим перегляду).
    def __init__(self, first_frame_timeout=10.0, low_latency=True, tello=None):
        t_start = time.time()
        self._tello = tello if tello is not None else Tello()

        # ── ЧОМУ СТАРТ ЗАЙМАВ 10-15 с І ЯК ЦЕ ПРИСКОРЕНО ────────────────────
        # djitellopy за замовчуванням чекає відповіді RESPONSE_TIMEOUT=7 с і
        # робить RETRY_COUNT=3 спроби. Тобто ОДНА команда без відповіді коштує
        # до 21 секунди. А в нас таких аж чотири джерела затримки:
        #   • connect: перший UDP-пакет часто губиться → 7 с у нікуди;
        #   • три налаштування відео — це команди SDK 2.0, і звичайний Tello
        #     (не EDU) на них просто МОВЧИТЬ → ще до 21 с кожна.
        # Рішення: на час ініціалізації тимчасово ставимо короткі таймаути, а
        # перед польотом ПОВЕРТАЄМО штатні (щоб takeoff/land мали повний запас).
        saved_timeout, saved_retry = Tello.RESPONSE_TIMEOUT, Tello.RETRY_COUNT
        try:
            Tello.RESPONSE_TIMEOUT = 2.0      # замість 7: втрачений пакет коштує 2 с
            self._tello.connect()             # UDP-рукостискання (192.168.10.1:8889)

            # Заряд — питання безпеки: нижче ~20% Tello може відмовитись злітати.
            battery = self._tello.get_battery()
            print(f"[tello] battery: {battery}%")
            if battery < 20:
                print("[tello] УВАГА: заряд низький, для польоту треба >20%")

            # Налаштування відео ДО streamon. Вони НЕОБОВ'ЯЗКОВІ, тож даємо їм
            # зовсім мало часу й ЖОДНОГО ретраю: не відповіли — йдемо далі.
            if low_latency:
                Tello.RESPONSE_TIMEOUT = 0.8
                Tello.RETRY_COUNT = 1
                self._try_video_setting("роздільна здатність 480p",
                                        self._tello.set_video_resolution, Tello.RESOLUTION_480P)
                self._try_video_setting("30 FPS",
                                        self._tello.set_video_fps, Tello.FPS_30)
                self._try_video_setting("бітрейт 3 Мбіт/с",
                                        self._tello.set_video_bitrate, Tello.BITRATE_3MBPS)
        finally:
            # ОБОВ'ЯЗКОВО: повертаємо штатні таймаути для польотних команд.
            Tello.RESPONSE_TIMEOUT, Tello.RETRY_COUNT = saved_timeout, saved_retry

        self._tello.streamon()                        # увімкнути відеопотік
        self._reader = self._tello.get_frame_read()   # фоновий читач кадрів
        self._t_start = t_start

        # Чекаємо на перший НЕ-чорний кадр (див. пастку 2 вище).
        deadline = time.time() + first_frame_timeout
        while time.time() < deadline:
            frame = self._reader.frame
            if frame is not None and frame.any():     # .any() = не всі пікселі нулі
                h, w = frame.shape[:2]
                # Друкуємо ЗАГАЛЬНИЙ час старту — щоб було видно, чи є прогрес,
                # і на що витрачається час, якщо колись знову стане повільно.
                print(f"[tello] відео пішло: {w}x{h}  (старт {time.time() - self._t_start:.1f} c)")
                return
            time.sleep(0.05)

        # Не дочекались — гасимо потік і кажемо ЧОМУ (щоб не ловити чорний екран).
        self._tello.streamoff()
        raise RuntimeError(
            f"Відео з Tello не з'явилось за {first_frame_timeout:.0f} с. "
            "Перевір: підключення до Wi-Fi TELLO-XXXXXX, вимкнений VPN, "
            "дозволений UDP у фаєрволі, актуальна прошивка дрона."
        )

    # Кожне налаштування — окремо й терпимо до помилки: якщо прошивка команди
    # не знає, потік однаково має піти (просто в дефолтному 720p).
    @staticmethod
    def _try_video_setting(label, setter, value):
        try:
            setter(value)
            print(f"[tello] {label}: ok")
        except Exception as e:
            print(f"[tello] {label}: не підтримується ({type(e).__name__}) — лишаємо як є")

    def read(self):
        frame = self._reader.frame
        if frame is None or not frame.any():
            return None                                # потік обірвався
        # RGB (від PIL) → BGR (як очікує OpenCV і весь наш конвеєр).
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        # Прибираємо за собою: без streamoff дрон продовжує слати UDP-потік.
        try:
            self._tello.streamoff()
        finally:
            self._tello.end()
