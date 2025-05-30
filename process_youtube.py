import cv2
import torch
import numpy as np
import time
import yt_dlp
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import os
import base64
from flask import Flask, request, jsonify, render_template, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import queue
import logging
import subprocess
import tempfile
import ffmpeg

# Initialize Flask and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for streaming
stream_queue = queue.Queue(maxsize=10)
is_streaming = False
current_stream = None
current_cap = None
current_process = None

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

def get_youtube_stream_url(youtube_url, quality='720p'):
    try:
        logger.info(f"Getting stream URL for: {youtube_url} with quality: {quality}")
        cmd = [
            "yt-dlp",
            "-g",
            "-f", f"bestvideo[height<={quality[:-1]}]+bestaudio/best[height<={quality[:-1]}]",
            youtube_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr}")
            raise Exception(f"Failed to get stream URL: {result.stderr}")
            
        stream_url = result.stdout.strip()
        logger.info(f"Successfully got stream URL: {stream_url}")
        return stream_url
    except Exception as e:
        logger.error(f"Error getting stream URL: {str(e)}")
        raise

def encode_frame(frame):
    """Encode frame to base64 for streaming"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')

def process_frame(frame):
    try:
        # Log frame information
        logger.info(f"Processing frame with shape: {frame.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        # Convert back to BGR for display
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        logger.info(f"Processed frame shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame

def stream_processor():
    global is_streaming, current_cap, current_process
    
    while True:
        if not is_streaming:
            time.sleep(0.1)
            continue
            
        try:
            if current_process is None:
                logger.error("FFmpeg process is not available")
                is_streaming = False
                socketio.emit('error', {'message': 'Failed to open video stream'})
                continue

            # Read frame from FFmpeg process
            frame_data = current_process.stdout.read(1920 * 1080 * 3)  # Read raw frame data
            if not frame_data:
                logger.error("Failed to read frame from FFmpeg")
                is_streaming = False
                socketio.emit('error', {'message': 'Failed to read video frame'})
                continue

            # Convert raw frame data to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((1080, 1920, 3))  # Reshape to video dimensions
            
            logger.info(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

            # Process the frame
            processed_frame = process_frame(frame)
            
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame through Socket.IO
            socketio.emit('video_frame', {
                'frame': frame_bytes,
                'timestamp': time.time()
            })
            
            logger.info("Frame sent successfully")
            
        except Exception as e:
            logger.error(f"Error in stream processor: {str(e)}")
            is_streaming = False
            socketio.emit('error', {'message': f'Streaming error: {str(e)}'})
            break

def process_youtube_stream(youtube_url, output_path=None, quality='720p', browser='chrome', cookies_file=None):
    """
    Process YouTube stream and either save to file or stream via Socket.IO
    
    Args:
        youtube_url (str): The YouTube URL to process
        output_path (str, optional): If provided, save processed video to this path
        quality (str): Video quality (e.g., '720p', '1080p')
        browser (str): Browser to use for cookies (default: 'chrome')
        cookies_file (str, optional): Path to cookies file
    
    Returns:
        bool: True if processing started successfully, False otherwise
    """
    global is_streaming, current_cap, current_stream, current_process
    
    try:
        logger.info(f"Starting YouTube stream processing for URL: {youtube_url}")
        
        # Get stream URL
        stream_url = get_youtube_stream_url(youtube_url, quality)
        if not stream_url:
            logger.error("Could not extract YouTube stream URL")
            return False
            
        logger.info(f"Successfully got stream URL: {stream_url}")
        
        # Stop any existing stream
        if current_process is not None:
            current_process.terminate()
            current_process = None
        
        # Start FFmpeg process to handle HLS stream
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', stream_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vf', 'scale=1280:720',  # Scale to 720p
            '-'
        ]
        
        logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
        
        current_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        if current_process.poll() is not None:
            error = current_process.stderr.read().decode()
            logger.error(f"FFmpeg process failed to start: {error}")
            return False
            
        current_stream = stream_url
        is_streaming = True
        
        # Start processing thread if not already running
        if not any(t.name == 'StreamProcessor' for t in threading.enumerate()):
            processor = threading.Thread(target=stream_processor, name='StreamProcessor')
            processor.daemon = True
            processor.start()
            
        logger.info("Stream processing started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in process_youtube_stream: {str(e)}")
        is_streaming = False
        if current_process is not None:
            current_process.terminate()
            current_process = None
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/youtube')
def youtube():
    return render_template('index.html')

@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    global is_streaming, current_cap, current_stream
    
    try:
        youtube_url = request.form.get('youtube_url')
        quality = request.form.get('quality', '720p')
        
        if not youtube_url:
            return jsonify({'error': 'No YouTube URL provided'}), 400
            
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        # Get stream URL
        stream_url = get_youtube_stream_url(youtube_url, quality)
        logger.info(f"Got stream URL: {stream_url}")
        
        # Stop any existing stream
        if current_cap is not None:
            current_cap.release()
        
        # Initialize new video capture
        current_cap = cv2.VideoCapture(stream_url)
        if not current_cap.isOpened():
            raise Exception("Failed to open video stream")
            
        current_stream = stream_url
        is_streaming = True
        
        # Start processing thread if not already running
        if not any(t.name == 'StreamProcessor' for t in threading.enumerate()):
            processor = threading.Thread(target=stream_processor, name='StreamProcessor')
            processor.daemon = True
            processor.start()
        
        return jsonify({
            'message': 'Stream started successfully',
            'stream_url': stream_url
        })
        
    except Exception as e:
        logger.error(f"Error processing YouTube stream: {str(e)}")
        is_streaming = False
        if current_cap is not None:
            current_cap.release()
            current_cap = None
        return jsonify({'error': str(e)}), 500

@socketio.on('stop_stream')
def stop_stream():
    global is_streaming, current_process
    logger.info("Stopping stream...")
    is_streaming = False
    if current_process is not None:
        current_process.terminate()
        current_process = None
    socketio.emit('processing_complete', {'message': 'Stream stopped'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_stream')
def handle_start_stream(data):
    youtube_url = data.get('url')
    if youtube_url:
        process_youtube_stream(youtube_url)
        return {'status': 'success', 'message': 'Streaming started'}
    return {'status': 'error', 'message': 'No YouTube URL provided'}

@socketio.on('stop_stream')
def handle_stop_stream():
    global is_streaming
    is_streaming = False
    return {'status': 'success', 'message': 'Streaming stopped'}

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 