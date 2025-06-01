import cv2
import subprocess
from flask import Flask, Response, render_template_string
from flask_cors import CORS
from pyngrok import ngrok
import cv2
import torch
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import subprocess
import os
import uuid
from datetime import datetime

# Area ROI dan parameter lain
area1 = [(830,280),(830,470),(90,470),(90,280)]
area2 = [(650,30),(650,140),(91,140),(91,30)]
CONF = 0.6
CLASS_ID = None
BLUR_ID = None


def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)
    # Top Left  x, y
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)
    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)
    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)
    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)
    return img

app = Flask(__name__)
CORS(app)

YOUTUBE_LIVE_URL = "https://www.youtube.com/watch?v=6QL0RHNtOlo"  # ganti sesuai live stream

def get_youtube_live_url(youtube_url):
    # Gunakan yt_dlp untuk dapatkan HLS m3u8 URL dari YouTube Live
    result = subprocess.run([
        "yt-dlp",
        "-g",               # Get direct video stream URL
        "-f", "best",
        youtube_url
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stream_url = result.stdout.strip()
    return stream_url

def generate_frames_from_stream(stream_url):
    count_v = 0
    vihicle_run_time = {}
    vihicle_in_roi = {}
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Gagal membuka stream")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print("Loading YOLO model...")
    tracker = DeepSort(max_age=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights='best.pt', device=device, fuse=True)
    model = AutoShape(model)

    with open('../configs/coco.names', "r") as f:
        class_names = f.read().strip().split("\n")
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    print("Starting video processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area in [area1, area2]:
            frame = cv2.polylines(frame, [np.array(area, np.int32)], True, (255,0,255), 6)
        cv2.putText(frame, "count : "+str(count_v),(50,50),0, 0.75, (255,255,255),2)

        results = model(frame)
        detect = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)
            if CLASS_ID is None:
                if confidence < CONF:
                    continue
            else:
                if class_id != CLASS_ID or confidence < CONF:
                    continue
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if BLUR_ID is not None and class_id == BLUR_ID:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

            centerX = (x1+x2)/2
            centerY = (y1+y2)/2
            result = cv2.pointPolygonTest(np.array(area2, np.int32),(int(centerX),int(centerY)), False)
            if result >= 0:
                vihicle_in_roi[track_id] = time.time()

            if track_id in vihicle_in_roi:
                result = cv2.pointPolygonTest(np.array(area1, np.int32),(int(centerX),int(centerY)), False)
                if result >= 0:
                    elapsed_time = time.time() - vihicle_in_roi[track_id]
                    if track_id not in vihicle_run_time:
                        vihicle_run_time[track_id] = elapsed_time
                        count_v += 1
                    if track_id in vihicle_run_time:
                        elapsed_time = vihicle_run_time[track_id]
                    distance = 50
                    speed_ms = distance / elapsed_time
                    speed_kh = speed_ms * 3.6
                    speed_txt = " Speed : " + str(int(speed_kh)) + "Km/h"
                    text = f"{track_id} - {class_names[class_id]} - {str(speed_txt)}"
                    frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head><title>YouTube Live Grayscale Stream</title></head>
    <body>
        <h1>Streaming YouTube Live (Grayscale)</h1>
        <img src="/video_feed" width="640" />
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    try:
        stream_url = get_youtube_live_url(YOUTUBE_LIVE_URL)
        return Response(generate_frames_from_stream(stream_url),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return f"Error loading stream: {str(e)}"

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * Ngrok URL: {public_url}")
    app.run(host="0.0.0.0", port=5000)
