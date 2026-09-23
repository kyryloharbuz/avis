
# РЕПЛЕЙ-СТЕНД: записати політ один раз — і програвати його замість живої
# камери скільки завгодно, тюнячи коефіцієнти на ІДЕНТИЧНОМУ вході.
#
# Навіщо: у реальному польоті кожен раз інший (ціль, світло, вітер), тож не
# зрозуміти, покращення від нового kp чи просто політ був легший. Реплей дає
# КОНТРОЛЬОВАНИЙ експеримент: той самий вхід, змінюєш одне — бачиш чистий ефект.
#
# ВАЖЛИВО: зберігаємо ВХІД пайплайну (кадри + детекції + dt), а не ВИХІД
# (команди). Команди перерахуються при відтворенні з новими коефіцієнтами —
# у цьому весь сенс.
#
# Формат запису (у теці recordings/, вона в .gitignore):
#   NAME.mp4    — самі кадри;
#   NAME.jsonl  — по рядку на кадр: {frame, dt, target_id, detections[]}.
# Recorder і ReplaySource живуть в одному файлі навмисно — щоб схема даних
# була в одному місці й не розсинхронізувалась.
#
# ReplaySource — це ТРЕТЯ реалізація FrameSource (поряд з Webcam і Tello). App
# не змінюється: він просто читає кадри. А два додаткові методи (replayed_dt,
# replayed_detections) App використовує, ЯКЩО вони є (duck typing) — так
# вмикається "лог-реплей": детекції беруться з файлу, YOLO не запускається,
# і тюнінг PID/Kalman летить у сотні разів швидше за реальний час.

import json
import os

import cv2

from avis.models import Detection
from avis.perception.frame_source import FrameSource


class FlightRecorder:
    """Пише кадри (.mp4) і вхід пайплайну (.jsonl) під час живого прогону."""

    def __init__(self, name, out_dir="recordings", fps=30):
        os.makedirs(out_dir, exist_ok=True)
        self._video_path = os.path.join(out_dir, f"{name}.mp4")
        self._log_path = os.path.join(out_dir, f"{name}.jsonl")
        self._fps = fps
        self._writer = None                 # лінива ініціалізація: розмір знаємо з 1-го кадру
        self._log = open(self._log_path, "w")
        self._frame_idx = 0
        print(f"[rec] запис у {self._video_path} + {self._log_path}")

    def write(self, frame, dt, detections, target_id,
              sent=None, state=None, battery=None):
        """Записати один кадр. Викликати з ЧИСТИМ кадром (до малювання рамок),
        інакше при реплеї рамки будуть "вигорілі" у відео.

        sent/state/battery — БАЗОВА ЛІНІЯ того польоту: які команди реально
        пішли в мотори, у якому стані був supervisor, який був заряд. При
        реплеї команди перераховуються заново, а ці — щоб було з ЧИМ
        порівняти ("стало різкіше — на скільки саме?")."""
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._video_path, fourcc, self._fps, (w, h))
        self._writer.write(frame)
        h, w = frame.shape[:2]
        record = {
            "frame": self._frame_idx,
            "dt": round(dt, 5),
            "target_id": target_id,
            # РОЗМІР КАДРУ. Потрібен офлайн-аналізу (score.py), який читає лише
            # .jsonl і відео не відкриває. Без нього доводилось припускати
            # 960x720, тоді як Tello віддає 648x478, — і перевірка "ціль у
            # центральній третині" рахувалась по неіснуючому кадру.
            "frame_w": w,
            "frame_h": h,
            "detections": [
                {
                    "id": d.id,
                    "xyxy": [round(v, 1) for v in d.xyxy],
                    "conf": round(d.conf, 3),
                    "cls": d.cls,
                    "name": d.name,
                }
                for d in detections
            ],
            "state": state,        # GROUNDED / HOVER / AUTO на цьому кадрі
            "battery": battery,    # % (кеш, без зайвих запитів до дрона)
            # Команди, що ПІШЛИ В МОТОРИ в тому польоті (None = політ вимкнений).
            "sent": None if sent is None else {
                "yaw": round(sent.yaw, 1),
                "vertical": round(sent.vertical, 1),
                "forward": round(sent.forward, 1),
            },
        }
        self._log.write(json.dumps(record) + "\n")
        self._frame_idx += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
        self._log.close()
        print(f"[rec] збережено {self._frame_idx} кадрів")


class ReplaySource(FrameSource):
    """Джерело кадрів із запису. Плюс віддає записані dt/детекції/ціль."""

    def __init__(self, name, out_dir="recordings"):
        video = os.path.join(out_dir, f"{name}.mp4")
        log = os.path.join(out_dir, f"{name}.jsonl")
        if not (os.path.exists(video) and os.path.exists(log)):
            raise FileNotFoundError(
                f"Немає запису '{name}' у {out_dir}/ (потрібні {name}.mp4 і {name}.jsonl)"
            )
        self._cap = cv2.VideoCapture(video)
        with open(log) as f:
            self._log = [json.loads(line) for line in f if line.strip()]
        self._idx = -1
        self._current = None
        print(f"[replay] '{name}': {len(self._log)} кадрів")

    def read(self):
        ok, frame = self._cap.read()
        nxt = self._idx + 1
        if not ok or nxt >= len(self._log):
            return None                      # кінець запису
        self._idx = nxt
        self._current = self._log[self._idx]  # рядок логу цього кадру
        return frame

    # --- Ці три методи App використовує ТІЛЬКИ у режимі реплею (через hasattr) ---
    def replayed_dt(self):
        """dt, ЯКИЙ БУВ У ПОЛЬОТІ. Критично: якщо при реплеї міряти реальний
        час між кадрами, він буде крихітний (женемо швидко) — і Kalman/PID
        зламаються. Тому віддаємо збережений dt, і фільтр 'думає', що час іде
        як у польоті, хоч насправді все летить у рази швидше."""
        return self._current["dt"]

    def replayed_target_id(self):
        """Яку ціль оператор вів у цьому кадрі (для відтворення вибору — кліку
        мишею при реплеї немає)."""
        return self._current.get("target_id")

    def baseline(self):
        """Команди/стан ТОГО польоту (базова лінія). None, якщо запис старого
        формату або політ був вимкнений. Потрібно, щоб порівнювати "було/стало"
        на однаковому вході."""
        return {
            "sent": self._current.get("sent"),
            "state": self._current.get("state"),
            "battery": self._current.get("battery"),
        }

    def replayed_detections(self):
        """Записані детекції цього кадру як об'єкти Detection (YOLO не запускаємо)."""
        return [
            Detection(
                id=d["id"], xyxy=tuple(d["xyxy"]), conf=d["conf"],
                cls=d["cls"], name=d.get("name", ""),
            )
            for d in self._current["detections"]
        ]

    def release(self):
        self._cap.release()
