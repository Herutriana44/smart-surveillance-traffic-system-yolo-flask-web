import os
import re
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from process_video import process_video
from process_youtube import process_youtube_stream
from pyngrok import ngrok
from datetime import datetime
import uuid

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


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

if __name__ == '__main__':
    # Only for Colab: start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print(f' * ngrok tunnel: {public_url}')
    app.run(port=5000) 