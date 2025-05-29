# Smart Surveillance Traffic System (SSTS)

Aplikasi web Flask untuk pemrosesan video traffic surveillance menggunakan YOLOv9 dan DeepSORT. Aplikasi ini dapat dijalankan di Google Colab dengan bantuan ngrok untuk port forwarding.

## Fitur

- Upload video untuk diproses
- Deteksi dan tracking kendaraan menggunakan YOLOv9 dan DeepSORT
- Perhitungan kecepatan kendaraan
- Preview hasil pemrosesan video
- Download video hasil pemrosesan

## Cara Menjalankan di Google Colab

### 1. Setup Environment

Buka Google Colab dan buat notebook baru. Jalankan perintah berikut untuk mengatur environment:

```python
# Clone repositori YOLOv9 dan DeepSORT
!git clone https://github.com/sujanshresstha/YOLOv9_DeepSORT.git
%cd YOLOv9_DeepSORT
!pip install -q -r requirements.txt

!git clone https://github.com/WongKinYiu/yolov9.git
%cd yolov9
!pip install -q -r requirements.txt

# Install dependensi tambahan
!pip install flask pyngrok deep_sort_realtime
```

### 2. Download Model Weights

```python
# Buat direktori weights dan download model
!mkdir -p weights
!wget -P weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
```

### 3. Upload File Aplikasi

Upload file-file berikut ke Google Colab:
- `app.py`
- `process_video.py`
- Folder `templates/` (berisi `index.html` dan `result.html`)

### 4. Jalankan Aplikasi

```python
# Jalankan aplikasi Flask
!python app.py
```

Setelah menjalankan aplikasi, Anda akan melihat output yang berisi URL ngrok. URL ini dapat digunakan untuk mengakses aplikasi web dari browser.

## Struktur File

```
.
├── app.py                 # Aplikasi Flask utama
├── process_video.py       # Fungsi pemrosesan video
├── templates/
│   ├── index.html        # Halaman upload
│   └── result.html       # Halaman hasil
├── uploads/              # Folder untuk video yang diupload
└── static/
    └── processed/        # Folder untuk video hasil proses
```

## Catatan Penting

1. Pastikan semua file berada di direktori yang benar di Google Colab.
2. File model YOLOv9 (`yolov9-c.pt`) harus ada di folder `weights/`.
3. File `coco.names` harus ada di folder `configs/`.
4. Proses video bisa memakan waktu tergantung ukuran video dan spesifikasi Colab.
5. URL ngrok akan berubah setiap kali menjalankan aplikasi.

## Troubleshooting

1. Jika terjadi error "No module named X":
   - Pastikan semua dependensi terinstall dengan benar
   - Jalankan `!pip install X` untuk menginstall modul yang kurang

2. Jika video tidak bisa diputar:
   - Pastikan format video yang diupload didukung (mp4, avi, mov)
   - Cek ukuran file tidak melebihi batas upload

3. Jika ngrok tidak berfungsi:
   - Pastikan pyngrok terinstall dengan benar
   - Coba jalankan ulang cell yang menjalankan aplikasi

## Kontribusi

Silakan buat issue atau pull request untuk kontribusi pada proyek ini. 