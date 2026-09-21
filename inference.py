
import cv2
import math
import cvzone
import requests
import geocoder
import os
import glob
import time
import re
from datetime import datetime
import pytz
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from ultralytics import YOLO
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

# --- Streaming imports ---
from flask import Flask, Response
import threading

# ---------------- Load Environment Variables ----------------
load_dotenv("/home/ubuntu/ShieldVision_assets/.env")
from_email = os.getenv("EMAIL_FROM")
password = os.getenv("EMAIL_PASS")
to_email = os.getenv("EMAIL_TO")
whatsapp_token = os.getenv("WHATSAPP_TOKEN")
client_secret_path = os.getenv("CLIENT_SECRET_PATH")
whatsapp_phone_id = os.getenv("WHATSAPP_PHONE_ID")
whatsapp_to = os.getenv("WHATSAPP_TO")

assets_folder = "/home/ubuntu/ShieldVision_assets"

PKT = pytz.timezone("Asia/Karachi")

# ---------------- Google Drive Authentication ----------------
gauth = GoogleAuth()
gauth.LoadClientConfigFile(client_secret_path)
cred_file = "/home/ubuntu/ShieldVision_assets/mycreds.txt"
gauth.LoadCredentialsFile(cred_file)
if gauth.credentials is None:
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()
gauth.SaveCredentialsFile(cred_file)
drive = GoogleDrive(gauth)

# ---------------- Load YOLO Model ----------------
model = YOLO("/home/ubuntu/ShieldVision_assets/best.pt")
classNames = ['gun']

# ---------------- Capture Video ----------------
cap = cv2.VideoCapture('rtmp://13.60.228.142/live/stream')

# ---------------- Location Functions ----------------
def get_location():
    g = geocoder.ip('me')
    return g.latlng if g.latlng else (None, None)

def get_address(lat, lng):
    if lat and lng:
        return f"https://www.google.com/maps?q={lat},{lng}"
    return "Location not available"

# ---------------- Upload Frame to Google Drive ----------------
def upload_to_drive(image_path):
    try:
        unique_name = os.path.basename(image_path)
        file = drive.CreateFile({'title': unique_name})
        file.SetContentFile(image_path)
        file.Upload()
        file.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
        return file['alternateLink']
    except Exception as e:
        print(f"❌ Google Drive upload failed: {e}")
        return None

# ---------------- Send WhatsApp Alert ----------------
def send_whatsapp_alert(phone_number, image_url, location_url, time, date):
    url = f"https://graph.facebook.com/v21.0/{whatsapp_phone_id}/messages"
    headers = {
        'Authorization': whatsapp_token,
        'Content-Type': 'application/json'
    }
    # Image
    image_payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "image",
        "image": {"link": image_url}
    }
    requests.post(url, json=image_payload, headers=headers)
    # Text
    text_payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "body": f"⚠ Security Alert: Gun Detected\n\n📍 Location: {location_url}\n⏰ Time: {time}\n📅 Date: {date}\n📷 Evidence: {image_url}\n\n*Immediate action required!*"
        }
    }
    requests.post(url, json=text_payload, headers=headers)

# ---------------- Send Email Alert ----------------
def send_email_alert(from_email, password, to_email, subject, body, attachment_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        msg.set_content(body)
        with open(attachment_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(attachment_path))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_email, password)
            smtp.send_message(msg)
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ---------------- Delete Old Files (>2 days) ----------------
def delete_old_files_by_days(folder_path, max_age_days=2):
    current_time = time.time()
    max_age_seconds = max_age_days * 86400
    for file in glob.glob(os.path.join(folder_path, "*.jpg")):
        if os.path.getctime(file) < (current_time - max_age_seconds):
            try:
                os.remove(file)
            except Exception as e:
                print(f"❌ Deletion error: {e}")

# ---------------- MJPEG streaming setup ----------------
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- Main detection loop ----------------
def run_detection():
    global output_frame, cap, lock
    while True:
        success, img = cap.read()
        if not success:
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture('rtmp://13.60.228.142/live/stream')
            continue

        # Resize for faster processing
        img_resized = cv2.resize(img, (640, 360))

        results = model(img_resized, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label = classNames[cls]

                cvzone.cornerRect(img_resized, (x1, y1, w, h), colorR=(255, 0, 0))
                cvzone.putTextRect(img_resized, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                if label == 'gun':
                    now = datetime.now(PKT)
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(assets_folder, f"detected_{timestamp}.jpg")
                    cv2.imwrite(image_path, img_resized)

                    delete_old_files_by_days(assets_folder, max_age_days=2)

                    drive_link = upload_to_drive(image_path)
                    if drive_link:
                        match = re.search(r'/d/([a-zA-Z0-9_-]+)', drive_link)
                        file_id = match.group(1) if match else None
                        direct_image_url = f"https://drive.google.com/uc?export=view&id={file_id}" if file_id else drive_link

                        lat, lng = get_location()
                        location_url = get_address(lat, lng)
                        time_str = now.strftime("%H:%M:%S")
                        date_str = now.strftime("%Y-%m-%d")

                        for number in whatsapp_to.split(','):
                            send_whatsapp_alert(number.strip(), direct_image_url, location_url, time_str, date_str)

                        subject = "🚨 Gun Detected Alert"
                        body = f"""🚨 A gun has been detected!

📍 Location: {location_url}
🕒 Time: {time_str}
📅 Date: {date_str}
📷 Image: {drive_link}

Immediate action is recommended.
"""
                        send_email_alert(from_email, password, to_email, subject, body, image_path)

        with lock:
            output_frame = img_resized.copy()

# ---------------- Start Flask + detection ----------------
if __name__ == "__main__":
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    time.sleep(1)
    run_detection()
