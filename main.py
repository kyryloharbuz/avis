from ultralytics import YOLO
import cv2, time

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = time.time()
    results = model.track(frame, imgsz=320, persist=True,
                          tracker="bytetrack.yaml", verbose=False)
    fps = 1 / (time.time() - t)

    h, w = frame.shape[:2]
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None and len(boxes) > 0:
        x1, y1, x2, y2 = boxes.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        error_x = cx - w / 2
        error_y = cy - h / 2
        area = (x2 - x1) * (y2 - y1)
        print(f"ex={error_x:+7.0f}  ey={error_y:+7.0f}  area={area:8.0f}")

    annotated = results[0].plot()
    cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("avis", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if cv2.getWindowProperty("avis", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()