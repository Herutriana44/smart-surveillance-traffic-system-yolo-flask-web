import os
import re
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from process_video import process_video
from process_youtube import process_youtube_stream
from pyngrok import ngrok
from datetime import datetime
import uuid
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import numpy as np
import time
import base64
import threading
import queue
import requests
import logging

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi NGROK URLs
TRAFFIC_DETECTION_URL = "1TfbVOS48SeXdQ7rJ2do5JjJFxG_4d5K3jMerctfbUsXvidrT"
VEHICLE_COUNTING_URL = "2xo4OVp1ka6HbnzvVq8dYvNQFZ6_6x7rCJyg45G4KaB3nt2Hd"

# Konfigurasi kamera
CAMERA_URL = "http://192.168.1.108:4747/video"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Konfigurasi antrian frame
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue(maxsize=30)

# Konfigurasi deteksi
DETECTION_INTERVAL = 1.0  # Interval deteksi dalam detik
last_detection_time = 0
last_count_time = 0

# Variabel global untuk menyimpan hasil
current_traffic_status = "Normal"
current_vehicle_count = 0
current_vehicle_types = {}

# Queue for video processing
video_queue = queue.Queue()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
        r'youtube\.com\/embed\/([^&\n?]+)',
        r'youtube\.com\/v\/([^&\n?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def process_video_stream(video_path, processed_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = process_video(frame)
        
        # Write processed frame
        out.write(processed_frame)
        
        # Convert frame to base64 for streaming
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Emit frame to connected clients
        socketio.emit('video_frame', {
            'frame': frame_base64,
            'progress': (frame_number / frame_count) * 100
        })
        
        frame_number += 1
        time.sleep(1/fps)  # Maintain original video speed
    
    cap.release()
    out.release()
    
    # Emit completion event
    socketio.emit('processing_complete', {
        'video_url': f'/processed/{os.path.basename(processed_path)}'
    })

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Process video and get unique output filename
            success, output_filename = process_video(input_path, app.config['PROCESSED_FOLDER'])
            
            if success:
                return redirect(url_for('result', filename=output_filename))
            else:
                return render_template('index.html', error='Error processing video')
    return render_template('index.html')

@app.route('/youtube', methods=['GET', 'POST'])
def youtube_stream():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url', '')
        if not youtube_url:
            return render_template('youtube.html', error='Please enter a YouTube URL')
        
        youtube_id = extract_youtube_id(youtube_url)
        if not youtube_id:
            return render_template('youtube.html', error='Invalid YouTube URL')
        
        # Generate output filename based on YouTube ID
        output_filename = f'processed_youtube_{youtube_id}.mp4'
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Process YouTube stream
        success = process_youtube_stream(youtube_url, output_path)
        if not success:
            return render_template('youtube.html', error='Failed to process YouTube stream')
        
        processed_url = url_for('static', filename=f'processed/{output_filename}')
        return render_template('youtube.html', 
                             youtube_url=youtube_url,
                             youtube_id=youtube_id,
                             processed_url=processed_url)
    
    return render_template('youtube.html')

@app.route('/result/<filename>')
def result(filename):
    video_url = url_for('static', filename=f'processed/{filename}')
    download_url = url_for('download_file', filename=filename)
    return render_template('result.html', video_url=video_url, download_url=download_url)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    if 'youtube_url' not in request.form:
        return jsonify({'error': 'No YouTube URL provided'}), 400
    
    youtube_url = request.form['youtube_url']
    quality = request.form.get('quality', '720p')
    browser = request.form.get('browser', 'chrome')  # Default to Chrome
    
    if not youtube_url:
        return jsonify({'error': 'Empty YouTube URL'}), 400
    
    # Handle cookies file upload
    cookies_file = None
    if 'cookies_file' in request.files:
        cookies = request.files['cookies_file']
        if cookies.filename:
            # Save cookies file temporarily
            cookies_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cookies.txt')
            cookies.save(cookies_path)
            cookies_file = cookies_path
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(app.static_folder, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f'youtube_output_{timestamp}_{unique_id}.mp4'
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Process the YouTube stream
        success = process_youtube_stream(
            youtube_url=youtube_url,
            output_path=output_path,
            quality=quality,
            browser=browser,
            cookies_file=cookies_file
        )
        
        if success:
            # Return the URL for the processed video
            video_url = url_for('static', filename=f'output/{output_filename}')
            return jsonify({
                'success': True,
                'video_url': video_url,
                'message': 'Video processed successfully'
            })
        else:
            return jsonify({
                'error': 'Failed to process YouTube stream. Please check the URL and try again.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'error': f'Error processing video: {str(e)}'
        }), 500
    finally:
        # Clean up cookies file if it was uploaded
        if cookies_file and os.path.exists(cookies_file):
            try:
                os.remove(cookies_file)
            except:
                pass

@app.route('/upload', methods=['POST'])
def upload_file_socket():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_filename = f'processed_{filename}'
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        
        file.save(video_path)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_video_stream, 
                                args=(video_path, processed_path))
        thread.start()
        
        return jsonify({'message': 'Processing started'})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/processed/<filename>')
def processed_video(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

def capture_frames():
    """Fungsi untuk menangkap frame dari kamera"""
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    if not cap.isOpened():
        logger.error("Error: Tidak dapat membuka kamera")
        return
    
    logger.info("Kamera berhasil dibuka")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Gagal membaca frame dari kamera")
            time.sleep(1)
            continue
        
        # Resize frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Simpan frame ke antrian
        if not frame_queue.full():
            frame_queue.put(frame)
        
        time.sleep(1/FPS)
    
    cap.release()

def process_frames():
    """Fungsi untuk memproses frame dan mendeteksi traffic"""
    global last_detection_time, last_count_time, current_traffic_status, current_vehicle_count, current_vehicle_types
    
    while True:
        if frame_queue.empty():
            time.sleep(0.1)
            continue
        
        frame = frame_queue.get()
        current_time = time.time()
        
        # Deteksi traffic setiap DETECTION_INTERVAL detik
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            try:
                # Konversi frame ke format yang sesuai
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                
                # Kirim request ke API traffic detection
                traffic_response = requests.post(
                    TRAFFIC_DETECTION_URL,
                    files={'image': ('image.jpg', img_bytes, 'image/jpeg')},
                    timeout=5
                )
                
                if traffic_response.status_code == 200:
                    traffic_data = traffic_response.json()
                    current_traffic_status = traffic_data.get('status', 'Normal')
                    logger.info(f"Traffic status: {current_traffic_status}")
                else:
                    logger.error(f"Error traffic detection: {traffic_response.status_code}")
                
                # Kirim request ke API vehicle counting
                vehicle_response = requests.post(
                    VEHICLE_COUNTING_URL,
                    files={'image': ('image.jpg', img_bytes, 'image/jpeg')},
                    timeout=5
                )
                
                if vehicle_response.status_code == 200:
                    vehicle_data = vehicle_response.json()
                    current_vehicle_count = vehicle_data.get('total_vehicles', 0)
                    current_vehicle_types = vehicle_data.get('vehicle_types', {})
                    logger.info(f"Vehicle count: {current_vehicle_count}")
                else:
                    logger.error(f"Error vehicle counting: {vehicle_response.status_code}")
                
                last_detection_time = current_time
                last_count_time = current_time
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error API request: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
        
        # Simpan hasil ke antrian
        if not result_queue.full():
            result_queue.put({
                'frame': frame,
                'traffic_status': current_traffic_status,
                'vehicle_count': current_vehicle_count,
                'vehicle_types': current_vehicle_types
            })

def generate_frames():
    """Fungsi untuk menghasilkan frame untuk streaming"""
    while True:
        if result_queue.empty():
            time.sleep(0.1)
            continue
        
        result = result_queue.get()
        frame = result['frame']
        
        # Tambahkan informasi ke frame
        cv2.putText(frame, f"Traffic: {result['traffic_status']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles: {result['vehicle_count']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Tambahkan informasi jenis kendaraan
        y_offset = 110
        for vehicle_type, count in result['vehicle_types'].items():
            cv2.putText(frame, f"{vehicle_type}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Konversi frame ke format yang sesuai untuk streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify({
        'traffic_status': current_traffic_status,
        'vehicle_count': current_vehicle_count,
        'vehicle_types': current_vehicle_types
    })

if __name__ == '__main__':
    # Buat direktori untuk menyimpan log jika belum ada
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Mulai thread untuk menangkap frame
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Mulai thread untuk memproses frame
    process_thread = threading.Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()
    
    # Jalankan aplikasi Flask
    app.run(host='0.0.0.0', port=5000, debug=True)