import cv2
import subprocess
from flask import Flask, Response, render_template_string
from flask_cors import CORS
from pyngrok import ngrok

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
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Gagal membuka stream")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Pengolahan OpenCV sederhana: convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        ret, buffer = cv2.imencode('.jpg', gray_rgb)
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
