import cv2
import subprocess
from flask import Flask, Response, render_template_string, jsonify
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
import threading


# Area ROI dan parameter lain
area1 = [(997,280),(997,418),(661,418),(661,280)]
area2 = [(973,536),(973,823),(280,823),(280,536)]

area3 = [(1895,537),(1895,811),(1009,811),(1009,537)]
area4 = [(1479,283),(1479,411),(1004,411),(1004,283)]
CONF = 0.6
CLASS_ID = None
BLUR_ID = None
CONGESTION_THRESHOLD = 10  # Threshold untuk deteksi kemacetan

# Global variables untuk menyimpan data
traffic_data = {
    'current_vehicles': 0,
    'total_vehicles': 0,
    'is_congested': False,
    'traffic_status': 'AMAN',
    'accident_count': 0,
    'pothole_count': 0
}

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
    global traffic_data
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

    with open('coco.names', "r") as f:
        class_names = f.read().strip().split("\n")
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    print("Starting video processing...")

    # download frame as image for first time
    ret, frame = cap.read()
    # print original frame width and height
    # print(f"Original frame width: {frame.shape[1]}, height: {frame.shape[0]}")
    cv2.imwrite('frame.jpg', frame)

    while True:
        ret, frame = cap.read()
        # print original frame width and height
        # print(f"Original frame width: {frame.shape[1]}, height: {frame.shape[0]}")

        if not ret:
            break

        for area in [area1, area2, area3, area4]:
            frame = cv2.polylines(frame, [np.array(area, np.int32)], True, (255,0,255), 6)
        # cv2.putText(frame, "count : "+str(count_v),(50,50),0, 0.75, (255,255,255),2)

        results = model(frame)
        detect = []
        current_vehicles = 0  # Counter untuk kendaraan yang terlihat saat ini
        current_accidents = 0
        current_potholes = 0

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
            # Increment counter jika kendaraan terdeteksi
            if class_id in [2, 3, 4, 5, 7]:  # ID untuk bicycle, car, motorbike, bus, truck
                current_vehicles += 1
            # Count accidents
            elif class_names[class_id].lower() == 'accident':
                current_accidents += 1
                traffic_data['accident_count'] += 1
            # Count potholes
            elif class_names[class_id].lower() == 'pothole':
                current_potholes += 1

        # Update traffic data
        traffic_data['current_vehicles'] = current_vehicles
        traffic_data['is_congested'] = current_vehicles >= CONGESTION_THRESHOLD
        traffic_data['traffic_status'] = 'KECELAKAAN' if current_accidents > 0 else 'AMAN'
        traffic_data['pothole_count'] = current_potholes

        # Tampilkan informasi pada frame
        cv2.putText(frame, f"Kendaraan: {current_vehicles}", (50, 50), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {'MACET' if traffic_data['is_congested'] else 'LANCAR'}", 
                   (50, 80), 0, 0.75, (0, 0, 255) if traffic_data['is_congested'] else (0, 255, 0), 2)
        cv2.putText(frame, f"Kondisi: {traffic_data['traffic_status']}", 
                   (50, 110), 0, 0.75, (0, 0, 255) if traffic_data['traffic_status'] == 'KECELAKAAN' else (0, 255, 0), 2)
        cv2.putText(frame, f"Total Kecelakaan: {traffic_data['accident_count']}", (50, 140), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Lubang Jalan: {current_potholes}", (50, 170), 0, 0.75, (255, 255, 255), 2)

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

            result = cv2.pointPolygonTest(np.array(area3, np.int32),(int(centerX),int(centerY)), False)
            if result >= 0:
                vihicle_in_roi[track_id] = time.time()

            if track_id in vihicle_in_roi:
                result = cv2.pointPolygonTest(np.array(area4, np.int32),(int(centerX),int(centerY)), False)
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
    <head>
        <title>Smart Traffic Monitoring</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .video-container {
                text-align: center;
                margin-top: 20px;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }
            .status-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
            }
            .status-card {
                padding: 15px;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status-card h3 {
                margin: 0 0 10px 0;
                color: #333;
            }
            .status-value {
                font-size: 24px;
                font-weight: bold;
            }
            .status-aman { color: #28a745; }
            .status-kecelakaan { color: #dc3545; }
            .status-macet { color: #dc3545; }
            .status-lancar { color: #28a745; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Traffic Monitoring System</h1>
            <div class="video-container">
                <img src="/video_feed" width="1280" />
            </div>
            <div class="status-container">
                <div class="status-card">
                    <h3>Kondisi Lalu Lintas</h3>
                    <div id="traffic-status" class="status-value">Loading...</div>
                </div>
                <div class="status-card">
                    <h3>Status Lalu Lintas</h3>
                    <div id="congestion-status" class="status-value">Loading...</div>
                </div>
                <div class="status-card">
                    <h3>Jumlah Kendaraan</h3>
                    <div id="vehicle-count" class="status-value">Loading...</div>
                </div>
                <div class="status-card">
                    <h3>Total Kecelakaan</h3>
                    <div id="accident-count" class="status-value">Loading...</div>
                </div>
                <div class="status-card">
                    <h3>Lubang Jalan</h3>
                    <div id="pothole-count" class="status-value">Loading...</div>
                </div>
            </div>
        </div>
        <script>
            function updateStatus() {
                fetch('/traffic_data')
                    .then(response => response.json())
                    .then(data => {
                        // Update traffic condition (AMAN/KECELAKAAN)
                        document.getElementById('traffic-status').textContent = data.traffic_status;
                        document.getElementById('traffic-status').className = 
                            'status-value status-' + data.traffic_status.toLowerCase();
                        
                        // Update congestion status (LANCAR/MACET)
                        const congestionStatus = data.is_congested ? 'MACET' : 'LANCAR';
                        document.getElementById('congestion-status').textContent = congestionStatus;
                        document.getElementById('congestion-status').className = 
                            'status-value status-' + congestionStatus.toLowerCase();
                        
                        // Update other values
                        document.getElementById('vehicle-count').textContent = data.current_vehicles;
                        document.getElementById('accident-count').textContent = data.accident_count;
                        document.getElementById('pothole-count').textContent = data.pothole_count;
                    });
            }
            setInterval(updateStatus, 1000);
        </script>
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

@app.route('/traffic_data')
def get_traffic_data():
    return jsonify(traffic_data)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * Ngrok URL: {public_url}")
    app.run(host="0.0.0.0", port=5000)
