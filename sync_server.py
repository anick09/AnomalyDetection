import os
import json
import datetime
import hashlib
import sqlite3
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

# Constants
SYNC_DB_PATH = "sync_server2.db"
INSPECTION_DB_PATH = "inspection_system_new5.db"
LOGS_DIR = "jsonlogs"
LOG_ARCHIVE_DIRS = ["process_image_logs", "main_logs"]

# Ensure required directories exist
os.makedirs(LOGS_DIR, exist_ok=True)

# 🔹 Fetch Inspection Details
def fetch_inspection_details(inspection_id):
    """Fetch AI detections, user feedback, and false annotations for an inspection."""
    conn = sqlite3.connect(INSPECTION_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT camera_index, x_min, y_min, x_max, y_max, confidence
        FROM BoundingBox_AI WHERE inspection_id = ?
    """, (inspection_id,))
    bounding_boxes = cursor.fetchall()

    cursor.execute("""
        SELECT camera_index, reviewer_comment, Anomalies_found, Anomalies_missed
        FROM Reviewer_Feedback WHERE inspection_id = ?
    """, (inspection_id,))
    feedback = cursor.fetchall()

    cursor.execute("""
        SELECT camera_index, x_min, y_min, width, height, type, comment
        FROM False_Annotations WHERE inspection_id = ?
    """, (inspection_id,))
    false_annotations = cursor.fetchall()

    conn.close()
    return bounding_boxes, feedback, false_annotations


### 🔹 Utility Functions ###

# 🔹 Fetch Last Synced Inspection ID
def get_last_synced_inspection():
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT end_inspection_id FROM SyncHistory ORDER BY id DESC LIMIT 1")
    last_sync = cursor.fetchone()
    conn.close()
    return last_sync[0] if last_sync else None

# 🔹 Fetch New Inspections
def fetch_new_inspections(last_inspection_id):
    conn = sqlite3.connect(INSPECTION_DB_PATH)
    cursor = conn.cursor()
    query = "SELECT inspection_id, session_id, inspection_timestamp, image_path1, image_path2 FROM Inspection"
    if last_inspection_id:
        query += " WHERE inspection_id > ?"
        cursor.execute(query, (last_inspection_id,))
    else:
        cursor.execute(query)
    inspections = cursor.fetchall()
    conn.close()
    return inspections


# 🔹 Sync Log Archive Files
def sync_log_files():
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    
    for log_dir in LOG_ARCHIVE_DIRS:
        for log_file in os.listdir(log_dir):
            if log_file.endswith(".log"):
                file_path = os.path.join(log_dir, log_file)
                cursor.execute("SELECT file_name FROM LogFileSync WHERE file_name = ?", (log_file,))
                if cursor.fetchone() is None:
                    cursor.execute("INSERT INTO LogFileSync (file_name, file_path) VALUES (?, ?)", (log_file, file_path))

    conn.commit()
    conn.close()

# 🔹 Generate Checksum
def generate_checksum(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# 🔹 Generate JSON Logs
def generate_json_logs(last_inspection_id):
    if last_inspection_id is None:
        last_inspection_id = 0
    new_inspections = fetch_new_inspections(last_inspection_id)
    
    if not new_inspections:
        return None, {}, None, None  # No new data

    log_data = []
    image_map = {}  # Maps image name to inspection ID
    
    start_id = new_inspections[0][0]
    end_id = new_inspections[-1][0]
    
    for inspection in new_inspections:
        inspection_id, session_id, timestamp, img1, img2 = inspection
        bounding_boxes, feedback, false_annotations = fetch_inspection_details(inspection_id)

        log_entry = {
            "inspection_id": inspection_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "images": {"image1": img1, "image2": img2},
            "bounding_boxes": [
                {"camera": cam, "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "confidence": conf}
                for cam, x1, y1, x2, y2, conf in bounding_boxes
            ],
            "feedback": [
                {"camera": cam, "comment": comment, "found": found, "missed": missed}
                for cam, comment, found, missed in feedback
            ],
            "false_annotations": [
                {"camera": cam, "x_min": x1, "y_min": y1, "width": w, "height": h, "type": t, "comment": c}
                for cam, x1, y1, w, h, t, c in false_annotations
            ]
        }
        log_data.append(log_entry)
        if img1:
            image_map[img1] = inspection_id
        if img2:
            image_map[img2] = inspection_id

    log_file = os.path.join(LOGS_DIR, f"logs_{int(datetime.datetime.utcnow().timestamp())}.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    return log_file, image_map, start_id, end_id

# 🚀 Sync API
@app.get("/sync/")
def sync_data():

     """Triggers data sync and ensures logs & images are fully acknowledged."""
     last_inspection_id = get_last_synced_inspection()
     print(f"Last synced inspection ID: {last_inspection_id}")


     log_file, image_map, start_id, end_id = generate_json_logs(last_inspection_id)
     if not log_file:
        sync_log_files()
        return {"message": "No new data to sync"}


     sync_log_files()

     conn = sqlite3.connect(SYNC_DB_PATH)
     cursor = conn.cursor()
    
     try:
        conn.execute("BEGIN TRANSACTION;")

        cursor.execute("INSERT INTO LogFile (file_name, file_path) VALUES (?, ?)", 
                        (os.path.basename(log_file), log_file))
        cursor.execute("INSERT INTO InspectionAcknowledgment (file_name, file_type) VALUES (?, 'log')",
                        (os.path.basename(log_file),))

        
        for image_name, inspection_id in image_map.items():
            #insert image file into ImageFile table
            cursor.execute("INSERT INTO ImageFile (image_name, image_path) VALUES (?, ?)",
                        (os.path.basename(image_name), image_name))
            #insert image file into InspectionAcknowledgment table
            cursor.execute("INSERT INTO InspectionAcknowledgment (inspection_id, file_name, file_type) VALUES (?, ?, ?)",
                        (inspection_id, os.path.basename(image_name), 'image'))
            
        cursor.execute("INSERT INTO SyncHistory (sync_timestamp, start_inspection_id, end_inspection_id) VALUES (?, ?, ?)",
                        (datetime.datetime.utcnow(), start_id, end_id))
        
        conn.commit()
        conn.close()
        return {"message": "Sync initiated. Awaiting client acknowledgment.", "log_file": log_file, "image_list": list(image_map.keys())}
     except Exception as e:
        conn.rollback()
        return {"message": "Sync failed", "error": str(e)}

     finally:
        conn.close()

### 📥 Acknowledgment API ###

@app.post("/acknowledge/")
def acknowledge_download(file_name: str):
    """Client confirms successful download of a file."""
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("UPDATE InspectionAcknowledgment SET client_acknowledged = 1 WHERE file_name = ?", (file_name,))
    if cursor.rowcount == 0:
        conn.close()
        return {"message": "File not found or already acknowledged"}

    conn.commit()
    conn.close()
    return {"message": f"{file_name} acknowledged successfully"}

### 🔄 Resend Mechanism ###

def resend_missing_files():
    """Check for files not acknowledged & resend them."""
    while True:
        conn = sqlite3.connect(SYNC_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_name, file_type FROM InspectionAcknowledgment 
            WHERE client_acknowledged = 0 
            AND last_attempt <= datetime('now', '-1 day')
        """)
        missing_files = cursor.fetchall()

        for file_name, file_type in missing_files:
            print(f"Resending {file_type}: {file_name}")  # Replace with actual resend logic
            cursor.execute("UPDATE InspectionAcknowledgment SET last_attempt = CURRENT_TIMESTAMP WHERE file_name = ?", 
                           (file_name,))
        
        conn.commit()
        conn.close()

        time.sleep(3600)  # Run every hour

### 📄 Download Endpoints ###

@app.get("/download_log/{log_id}")
def download_log(log_id: int):
    """Download a log file."""
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM LogFile WHERE id = ?", (log_id,))
    log = cursor.fetchone()
    conn.close()
    if not log:
        return {"message": "Log file not found"}
    return FileResponse(log[0], filename=os.path.basename(log[0]))

@app.get("/download_image/{image_id}")
def download_image(image_id: int):
    """Download an image file."""
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM ImageFile WHERE id = ?", (image_id,))
    image = cursor.fetchone()
    conn.close()
    if not image:
        return {"message": "Image not found"}
    return FileResponse(image[0], filename=os.path.basename(image[0]))



# 📥 1. Get a List of All Unacknowledged Logs
@app.get("/list_logs/")
def list_logs():
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()

    # Fetch unacknowledged log file names and paths
    cursor.execute("SELECT file_name, file_path FROM LogFileSync WHERE client_acknowledged = 0 ORDER BY synced_at ASC")
    logs = cursor.fetchall()
    
    conn.close()

    if not logs:
        return JSONResponse(content={"message": "No logs available for download"}, status_code=404)

    # Convert results into a list of dictionaries
    log_list = [{"file_name": log[0], "file_path": log[1]} for log in logs]
    
    return {"logs": log_list}

# 📥 2. Download a Specific Log File by Name
@app.get("/download_log/")
def download_log(file_name: str):
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()

    # Fetch the file path based on file name
    cursor.execute("SELECT file_path FROM LogFileSync WHERE file_name = ? AND client_acknowledged = 0", (file_name,))
    log = cursor.fetchone()

    conn.close()

    if not log:
        raise HTTPException(status_code=404, detail="Log file not found or already acknowledged")

    file_path = log[0]

    # Validate if file exists on disk
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {file_name} not found on server")

    return FileResponse(file_path, filename=file_name, media_type="application/octet-stream")

# 📥 3. Acknowledge a Log File After Download
from pydantic import BaseModel

class AcknowledgeRequest(BaseModel):
    file_name: str

@app.post("/acknowledge_log/")
def acknowledge_log(request: AcknowledgeRequest):
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("UPDATE LogFileSync SET client_acknowledged = 1 WHERE file_name = ?", (request.file_name,))
    conn.commit()
    conn.close()

    return {"message": f"Log file {request.file_name} acknowledged successfully"}

