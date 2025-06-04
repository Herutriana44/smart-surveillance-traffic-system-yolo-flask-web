import cv2
import subprocess
from flask import Flask, Response, render_template, jsonify, request
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

# Dictionary untuk multiple live streams
YOUTUBE_LIVE_URLS = {
    "JL. PANTURA SAYUNG DEMAK [01] by DISHUB KAB. DEMAK": "https://www.youtube.com/watch?v=6QL0RHNtOlo",
    "JL. PANTURA SAYUNG DEMAK [02] by DISHUB KAB. DEMAK": "https://www.youtube.com/watch?v=6QL0RHNtOlo",
    "JL. PANTURA DEPAN PASAR BUYARAN DEMAK [01] by DISHUB KAB. DEMAK": "https://www.youtube.com/watch?v=asdJILkKNfs",
    "EXIT TOL JL. LINGKAR DEMAK [02] by DISHUB KAB. DEMAK": "https://www.youtube.com/watch?v=T04pR1ZIfkU",
    "EXIT TOL JL. LINGKAR DEMAK [01] by DISHUB KAB. DEMAK": "https://www.youtube.com/watch?v=_IFD0Ah8a-M",
}

# Global variables untuk menyimpan data setiap stream
traffic_data = {
    stream_name: {
        'current_vehicles': 0,
        'total_vehicles': 0,
        'is_congested': False,
        'traffic_status': 'AMAN',
        'accident_count': 0,
        'pothole_count': 0
    } for stream_name in YOUTUBE_LIVE_URLS.keys()
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

def get_youtube_live_url(youtube_url):
    result = subprocess.run([
        "yt-dlp",
        "-g",
        "-f", "best",
        youtube_url
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stream_url = result.stdout.strip()
    return stream_url

def generate_frames_from_stream(stream_url, stream_name):
    global traffic_data
    count_v = 0
    vihicle_run_time = {}
    vihicle_in_roi = {}
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Gagal membuka stream: {stream_name}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Loading YOLO model for {stream_name}...")
    tracker = DeepSort(max_age=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights='best.pt', device=device, fuse=True)
    model = AutoShape(model)

    with open('coco.names', "r") as f:
        class_names = f.read().strip().split("\n")
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    print(f"Starting video processing for {stream_name}...")

    ret, frame = cap.read()
    cv2.imwrite(f'frame_{stream_name}.jpg', frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area in [area1, area2, area3, area4]:
            frame = cv2.polylines(frame, [np.array(area, np.int32)], True, (255,0,255), 6)

        results = model(frame)
        detect = []
        current_vehicles = 0
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
            
            if class_id in [2, 3, 4, 5, 7]:  # bicycle, car, motorbike, bus, truck
                current_vehicles += 1
            elif class_names[class_id].lower() == 'accident':
                current_accidents += 1
                traffic_data[stream_name]['accident_count'] += 1
            elif class_names[class_id].lower() == 'pothole':
                current_potholes += 1

        # Update traffic data for this stream
        traffic_data[stream_name]['current_vehicles'] = current_vehicles
        traffic_data[stream_name]['is_congested'] = current_vehicles >= CONGESTION_THRESHOLD
        traffic_data[stream_name]['traffic_status'] = 'KECELAKAAN' if current_accidents > 0 else 'AMAN'
        traffic_data[stream_name]['pothole_count'] = current_potholes

        # Display information on frame
        cv2.putText(frame, f"{stream_name}", (50, 30), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Kendaraan: {current_vehicles}", (50, 60), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {'MACET' if traffic_data[stream_name]['is_congested'] else 'LANCAR'}", 
                   (50, 90), 0, 0.75, (0, 0, 255) if traffic_data[stream_name]['is_congested'] else (0, 255, 0), 2)
        cv2.putText(frame, f"Kondisi: {traffic_data[stream_name]['traffic_status']}", 
                   (50, 120), 0, 0.75, (0, 0, 255) if traffic_data[stream_name]['traffic_status'] == 'KECELAKAAN' else (0, 255, 0), 2)
        cv2.putText(frame, f"Total Kecelakaan: {traffic_data[stream_name]['accident_count']}", (50, 150), 0, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Lubang Jalan: {current_potholes}", (50, 180), 0, 0.75, (255, 255, 255), 2)

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
    return render_template('landing.html')

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html', stream_names=YOUTUBE_LIVE_URLS.keys())

@app.route('/single-stream')
def single_stream():
    stream_name = request.args.get('url')
    if stream_name not in YOUTUBE_LIVE_URLS:
        return "Stream not found", 404
    return render_template('single_stream.html', stream_name=stream_name)

@app.route('/video_feed/<stream_name>')
def video_feed(stream_name):
    try:
        if stream_name in YOUTUBE_LIVE_URLS:
            stream_url = get_youtube_live_url(YOUTUBE_LIVE_URLS[stream_name])
            return Response(generate_frames_from_stream(stream_url, stream_name),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return "Stream not found", 404
    except Exception as e:
        return f"Error loading stream: {str(e)}"

@app.route('/traffic_data')
def get_traffic_data():
    return jsonify(traffic_data)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * Ngrok URL: {public_url}")
    app.run(host="0.0.0.0", port=5000)
