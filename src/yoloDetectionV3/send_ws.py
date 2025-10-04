# send_ws.py
import cv2
import time
import websocket  # pip install websocket-client

WS_URL = "ws://localhost:8000/ingest/ws"
VIDEO_PATH = r"C:/Users/Gamer/PycharmProjects/soyplant-detect-api/data/videos/teste9.mp4"
TARGET_FPS = 30.0  # defina a taxa de envio

def main():
    ws = websocket.create_connection(WS_URL, timeout=5)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu: {VIDEO_PATH}")

    frame_interval = 1.0 / TARGET_FPS
    last = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            ws.send_binary(buf.tobytes())

            # simples controle de fps
            diff = time.time() - last
            if diff < frame_interval:
                time.sleep(frame_interval - diff)
            last = time.time()
    finally:
        cap.release()
        try: ws.close()
        except: pass

if __name__ == "__main__":
    main()
