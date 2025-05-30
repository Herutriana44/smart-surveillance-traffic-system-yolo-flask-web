from app import app, socketio
from pyngrok import ngrok
import threading
import time

def start_ngrok():
    # Tunggu sebentar untuk memastikan tidak ada tunnel yang aktif
    time.sleep(1)
    # Tutup semua tunnel yang mungkin masih aktif
    ngrok.kill()
    # Buat tunnel baru
    public_url = ngrok.connect(5000)
    print(f' * ngrok tunnel: {public_url}')
    return public_url

if __name__ == '__main__':
    # Start ngrok in a separate thread
    ngrok_thread = threading.Thread(target=start_ngrok)
    ngrok_thread.daemon = True
    ngrok_thread.start()
    
    # Run the Socket.IO server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)