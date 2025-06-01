import os
import re
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from process_video import process_video
from process_youtube import process_youtube
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
            
        # Process frame using the new process_video function
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
        success = process_youtube(youtube_url, output_path)
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
def handle_youtube_stream():
    try:
    if 'youtube_url' not in request.form:
            print("Error: No YouTube URL provided in request")
        return jsonify({'error': 'No YouTube URL provided'}), 400
    
    youtube_url = request.form['youtube_url']
    quality = request.form.get('quality', '720p')
    browser = request.form.get('browser', 'chrome')  # Default to Chrome
    
    if not youtube_url:
            print("Error: Empty YouTube URL")
        return jsonify({'error': 'Empty YouTube URL'}), 400
    
        print(f"Processing YouTube URL: {youtube_url}")
        print(f"Quality: {quality}")
        print(f"Browser: {browser}")
        
    # Handle cookies file upload
    cookies_file = None
    if 'cookies_file' in request.files:
        cookies = request.files['cookies_file']
        if cookies.filename:
            # Save cookies file temporarily
            cookies_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cookies.txt')
            cookies.save(cookies_path)
            cookies_file = cookies_path
                print(f"Using cookies file: {cookies_path}")
    
    try:
            # Process the YouTube stream without saving to file
            success = process_youtube(
            youtube_url=youtube_url,
            quality=quality,
            browser=browser,
            cookies_file=cookies_file
        )
        
        if success:
                print("YouTube stream processing started successfully")
            return jsonify({
                'success': True,
                    'message': 'Stream processing started successfully'
            })
        else:
                print("Failed to start YouTube stream processing")
                return jsonify({
                    'success': False,
                    'error': 'Failed to start stream processing'
                }), 500
                
        except Exception as e:
            print(f"Error processing stream: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error processing stream: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error in handle_youtube_stream: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            try:
                frame = video_queue.get_nowait()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except queue.Empty:
                # If queue is empty, send a blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # Add small delay to prevent high CPU usage

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start video stream from YouTube URL"""
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        return jsonify({'error': 'No YouTube URL provided'}), 400
    
    try:
        # Start processing in a separate thread
        thread = threading.Thread(target=process_youtube,
                                args=(youtube_url, None))
        thread.daemon = True
        thread.start()
        return jsonify({'message': 'Stream started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video stream"""
    # Clear the video queue
    while not video_queue.empty():
        try:
            video_queue.get_nowait()
        except queue.Empty:
            break
    return jsonify({'message': 'Stream stopped successfully'})

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current traffic status"""
    return jsonify({
        'traffic_status': 'Normal',  # Replace with actual status
        'vehicle_count': 0,  # Replace with actual count
        'vehicle_types': {}  # Replace with actual vehicle types
    })

if __name__ == '__main__':
    # Only for Colab: start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print(f' * ngrok tunnel: {public_url}')
    socketio.run(app, debug=False, host='0.0.0.0', port=5000) 