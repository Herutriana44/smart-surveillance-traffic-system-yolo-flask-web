import cv2
import torch
import numpy as np
import time
import yt_dlp
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import os
import base64
from flask import Flask, request, jsonify, render_template_string, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import queue
import logging
import subprocess
import tempfile

# Initialize Flask and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for streaming
is_streaming = False
current_process = None
current_cap = None

def get_youtube_stream_url(youtube_url, quality='720p'):
    try:
        logger.info(f"Getting stream URL for: {youtube_url} with quality: {quality}")
        ydl_opts = {
            'format': f'bestvideo[height<={quality[:-1]}]+bestaudio/best[height<={quality[:-1]}]',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if 'url' in info:
                stream_url = info['url']
                logger.info(f"Successfully got stream URL: {stream_url}")
                return stream_url
            else:
                logger.error("No URL found in video info")
                return None
                
    except Exception as e:
        logger.error(f"Error getting stream URL: {str(e)}")
        raise

def process_frame(frame):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        # Convert back to BGR for display
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame

def generate_frames():
    global is_streaming, current_process
    
    while is_streaming and current_process is not None:
        try:
            # Read frame from FFmpeg process
            frame_data = current_process.stdout.read(1280 * 720 * 3)  # Read raw frame data for 720p
            if not frame_data:
                logger.error("Failed to read frame from FFmpeg")
                break

            # Convert raw frame data to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((720, 1280, 3))  # Reshape to 720p dimensions
            
            # Process frame
            processed_frame = process_frame(frame)
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                logger.error("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Error in generate_frames: {str(e)}")
            break

def process_youtube(youtube_url, output_path=None, quality='720p', browser='chrome', cookies_file=None):
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
    global is_streaming, current_process
    
    try:
        if not youtube_url:
            logger.error("No YouTube URL provided")
            return False
            
        logger.info(f"Starting YouTube stream processing for URL: {youtube_url}")
        
        # Get stream URL
        stream_url = get_youtube_stream_url(youtube_url, quality)
        if not stream_url:
            logger.error("Could not extract YouTube stream URL")
            return False
            
        logger.info(f"Successfully got stream URL: {stream_url}")
        
        # Stop any existing stream
        is_streaming = False
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

        is_streaming = True
        
        # If output_path is provided, save the processed video
        if output_path:
            # Get video properties
            fps = 30  # Default FPS
            width = 1280
            height = 720
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while is_streaming:
                try:
                    # Read frame from FFmpeg process
                    frame_data = current_process.stdout.read(width * height * 3)
                    if not frame_data:
                        break

                    # Convert raw frame data to numpy array
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((height, width, 3))
                    
                    # Process frame
                    processed_frame = process_frame(frame)
                    
                    # Write processed frame
                    out.write(processed_frame)
                    
                except Exception as e:
                    logger.error(f"Error processing frame for output: {str(e)}")
                    break
                    
            out.release()
            
        logger.info("Stream processing started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in process_youtube: {str(e)}")
        is_streaming = False
        if current_process is not None:
            current_process.terminate()
            current_process = None
        return False

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>YouTube Live Stream Processing</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
            }
            .controls {
                margin: 20px 0;
                text-align: center;
            }
            input[type="text"] {
                width: 70%;
                padding: 8px;
                margin-right: 10px;
            }
            button {
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YouTube Live Stream Processing</h1>
            <div class="controls">
                <input type="text" id="youtube_url" placeholder="Enter YouTube URL">
                <button onclick="startStream()">Start Stream</button>
                <button onclick="stopStream()">Stop Stream</button>
            </div>
            <div class="video-container">
                <img src="/video_feed" width="640" />
            </div>
        </div>
        
        <script>
            function startStream() {
                const url = document.getElementById('youtube_url').value;
                if (!url) {
                    alert('Please enter a YouTube URL');
                    return;
                }
                
                fetch('/process_youtube', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `youtube_url=${encodeURIComponent(url)}`
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Stream started successfully');
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error starting stream');
                });
            }
            
            function stopStream() {
                fetch('/stop_stream', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Stream stopped successfully');
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error stopping stream');
                });
            }
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_youtube', methods=['POST'])
def handle_youtube_stream():
    global is_streaming, current_process
    
    try:
        if 'youtube_url' not in request.form:
            return jsonify({'error': 'No YouTube URL provided'}), 400
        
        youtube_url = request.form['youtube_url']
        quality = request.form.get('quality', '720p')
        
        if not youtube_url:
            return jsonify({'error': 'Empty YouTube URL'}), 400
        
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        # Process the stream
        success = process_youtube(youtube_url, quality=quality)
        if not success:
            return jsonify({'error': 'Failed to start stream processing'}), 500
            
        return jsonify({
            'success': True,
            'message': 'Stream started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in handle_youtube_stream: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global is_streaming, current_process
    
    try:
        is_streaming = False
        if current_process is not None:
            current_process.terminate()
            current_process = None
            
        return jsonify({
            'success': True,
            'message': 'Stream stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in stop_stream: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 