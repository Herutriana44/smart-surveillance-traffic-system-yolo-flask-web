import cv2
import torch
import numpy as np
import time
import yt_dlp
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import os

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

def get_youtube_stream_url(youtube_url, quality='720p', browser='chrome', cookies_file=None):
    """Extract direct stream URL from YouTube using yt-dlp with browser cookies"""
    print(f"🔍 Extracting stream URL from YouTube...")
    
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    
    # Define possible browser cookie paths
    cookie_paths = {
        'chrome': os.path.join(home_dir, 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'Network', 'Cookies'),
        'firefox': os.path.join(home_dir, 'AppData', 'Roaming', 'Mozilla', 'Firefox', 'Profiles'),
        'edge': os.path.join(home_dir, 'AppData', 'Local', 'Microsoft', 'Edge', 'User Data', 'Default', 'Network', 'Cookies')
    }
    
    ydl_opts = {
        'format': f'best[height<={quality[:-1]}]/best',
        'quiet': True,
        'no_warnings': True,
    }
    
    # Add cookies options
    if cookies_file and os.path.exists(cookies_file):
        print(f"Using cookies from file: {cookies_file}")
        ydl_opts['cookiefile'] = cookies_file
    else:
        print(f"Using cookies from browser: {browser}")
        ydl_opts['cookiesfrombrowser'] = (browser,)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if 'url' in info:
                stream_url = info['url']
                print(f"✅ Stream URL extracted successfully")
                print(f"   Title: {info.get('title', 'Unknown')}")
                return stream_url
            else:
                print("❌ Could not extract stream URL")
                return None
    except Exception as e:
        print(f"❌ Error extracting YouTube URL: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're logged into YouTube in your browser")
        print("2. Try using a different browser (chrome/firefox/edge)")
        print("3. If using Chrome, try closing all Chrome windows first")
        print("4. Check if your browser's cookies are accessible")
        print("5. If using cookies file, make sure it exists and is in Netscape format")
        return None

def process_youtube_stream(youtube_url, output_path=None, weights_path='./weights/yolov9-c.pt', 
                         classes_path='../configs/coco.names', quality='720p', max_frames=300,
                         browser='chrome', cookies_file=None):
    """
    Process YouTube stream and save to file
    max_frames: Maximum number of frames to process (default: 300 frames = ~10 seconds at 30fps)
    browser: Browser to use for cookies (chrome/firefox/edge)
    cookies_file: Path to cookies.txt file in Netscape format
    """
    count_v = 0
    vihicle_run_time = {}
    vihicle_in_roi = {}
    frame_count = 0

    # Get YouTube stream URL with browser cookies
    stream_url = get_youtube_stream_url(youtube_url, quality, browser, cookies_file)
    if not stream_url:
        return False

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Error: Could not open YouTube stream")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer
    if not output_path:
        output_path = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Loading YOLOv9 model...")
    tracker = DeepSort(max_age=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights=weights_path, device=device, fuse=True)
    model = AutoShape(model)

    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    print("Starting video processing...")
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Stream ended or connection lost")
                break

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{max_frames}")

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

            writer.write(frame)

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return False
    finally:
        cap.release()
        writer.release()
        print(f"✅ Processing completed. Saved {frame_count} frames to {output_path}")
    
    return True 