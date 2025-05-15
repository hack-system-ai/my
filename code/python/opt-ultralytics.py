import cv2
from ultralytics import YOLO
import threading
import queue
import time

frame_queue = queue.Queue(timeout=2)
result_queue = queue.Queue(timeout=2)

def capture_threading(cap, frame_queue):
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (320, 320))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

def inference_thread(model, frame_queue, result_queue):
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)

            results = model(frame, imgsz=320, conf=0.5, device='cpu')

            processed_results = process_results(results)

            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put(processed_results)
        except queue.Empty:
            time.sleep(0.01)
            continue

def process_results(results):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    return {
        'boxes': boxes,
        'classes': classes,
        'confidences': confidences
    }

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    model = YOLO('bast.pt')

    t_capture = threading.Thread(target=capture_threading, args=(cap, frame_queue), daemon=True)
    t_inference = threading.Thread(target=inference_thread, args=(model, frame_queue, result_queue), daemon=True)

    t_capture.start()
    t_inference.start()

    fps = 0
    last_time = time.time()

    try:
        while True:
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1 / (current_time - last_time))
            last_time = current_time

            result = None
            if not result_queue.empty():
                result = result_queue.get()

            if not frame_queue.empty():
                display_frame = frame_queue.queue[-1]
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if result is not None:
                    for box, cls_id, conf, in zip(result['boxes'], result['classes'], result['confidences']):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{model.names[int(cls_id)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('YOLO Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
                