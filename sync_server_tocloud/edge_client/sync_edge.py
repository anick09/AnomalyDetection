import os
import json
import datetime
import hashlib
import sqlite3
import requests

# Constants
SYNC_DB_PATH = "data/sync_server2.db"
INSPECTION_DB_PATH = "data/inspection_system_new5.db"
LOGS_DIR = "data/jsonlogs"
LOGS_DIR1 = "data/main_logs"
LOGS_DIR2 = "data/process_image_logs"
CLOUD_SERVER_URL = "http://192.168.1.91:8002/api/upload"
os.makedirs(LOGS_DIR, exist_ok=True)

# Fetch Inspection Details
def fetch_inspection_details(inspection_id):
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

# Get Last Synced Inspection ID
def get_last_synced_inspection():
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT end_inspection_id FROM SyncHistory ORDER BY id DESC LIMIT 1")
    last_sync = cursor.fetchone()
    conn.close()
    return last_sync[0] if last_sync else None

# Update Sync History
def update_sync_history(start_id, end_id):
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO SyncHistory (start_inspection_id, end_inspection_id, sync_timestamp)
        VALUES (?, ?, ?)
    """, (start_id, end_id, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# Fetch New Inspections Before Current Date
def fetch_new_inspections(last_inspection_id):
    conn = sqlite3.connect(INSPECTION_DB_PATH)
    cursor = conn.cursor()
    current_date = datetime.datetime.utcnow().date().isoformat()
    query = """
        SELECT inspection_id, session_id, inspection_timestamp, image_path1, image_path2 
        FROM Inspection
        WHERE inspection_id > ? AND DATE(inspection_timestamp) < ?
    """
    cursor.execute(query, (last_inspection_id, current_date))
    inspections = cursor.fetchall()
    conn.close()
    return inspections

# Generate JSON Logs
def generate_json_logs(last_inspection_id):
    if last_inspection_id is None:
        last_inspection_id = 0
    new_inspections = fetch_new_inspections(last_inspection_id)
    print(len(new_inspections))

    if not new_inspections:
        return None, {}, None, None

    log_data = []
    image_map = {}

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

# Generate Checksum
def generate_checksum(file_path):
    sha256_hash = hashlib.sha256()
    print(f"Generating checksum for: {file_path}")
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# # Function to Log Synced Images
# def log_synced_image(file_name, checksum, status):
#     conn = sqlite3.connect(SYNC_DB_PATH)
#     cursor = conn.cursor()
#     timestamp = datetime.datetime.utcnow().isoformat()
#     cursor.execute("""
#         INSERT INTO ImageSyncHistory (file_name, checksum, sync_timestamp, status)
#         VALUES (?, ?, ?, ?)
#     """, (file_name, checksum, timestamp, status))
#     conn.commit()
#     conn.close()
# Function to Log Synced Files

def log_synced_file(file_name, checksum, status, file_type):
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO FileSyncHistory (file_name, checksum, sync_timestamp, status, file_type)
        VALUES (?, ?, ?, ?, ?)
    """, (file_name, checksum, timestamp, status, file_type))
    conn.commit()
    conn.close()

# Fetch Unsynced Log Files Before Current Date
def fetch_unsynced_log_files():
    conn = sqlite3.connect(SYNC_DB_PATH)
    cursor = conn.cursor()
    current_date = datetime.datetime.utcnow().date().isoformat()
    cursor.execute("""
        SELECT file_name FROM FileSyncHistory
        WHERE DATE(sync_timestamp) < ? AND status != 'success'
    """, (current_date,))
    synced_files = {row[0] for row in cursor.fetchall()}
    conn.close()

    unsynced_files = []
    for log_dir in [LOGS_DIR1, LOGS_DIR2]:
        for f in os.listdir(log_dir):
            file_path = os.path.join(log_dir, f)
            # file_date = datetime.datetime.utcfromtimestamp(os.path.getmtime(file_path)).date().isoformat()
            file_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path), datetime.UTC).date().isoformat()
            if (f.endswith('.log') and os.path.isfile(file_path)
                    and f not in synced_files and file_date < current_date):
                unsynced_files.append(file_path)

    return unsynced_files

def upload_to_cloud(log_file, image_map, unsynced_logs):
    # Generate checksums
    checksums = {"log_file": generate_checksum(log_file)}
    files = [("files", (os.path.basename(log_file), open(log_file, "rb")))]
    
    print(f"Checksum: {checksums['log_file']}")
    
    # Add image files
    for path, inspection_id in image_map.items():
        if os.path.exists(path):
            print(f"Uploading image: {path}")
            files.append(("files", (os.path.basename(path), open(path, "rb"))))
            checksums[os.path.basename(path)] = generate_checksum(path)
        else:
            print(f"Image file not found: {path}")
    
    # Add unsynced logs
    for log_path in unsynced_logs:
        if os.path.exists(log_path):
            print(f"Uploading unsynced log: {log_path}")
            files.append(("files", (os.path.basename(log_path), open(log_path, "rb"))))
            checksums[os.path.basename(log_path)] = generate_checksum(log_path)
    
    # Send files with checksums
    response = requests.post(
        CLOUD_SERVER_URL,
        files=files,
        data={"checksums": json.dumps(checksums)}
    )

    # Close file objects
    for _, file_obj in files:
        file_obj[1].close()

    # Handle response
    if response.status_code == 200:
        try:
            response_data = response.json()
            if response_data.get("status") == "success":
                server_checksums = response_data.get("checksums", {})
                # Verify and log each file individually
                for file_name, checksum in checksums.items():
                    file_type = (
                        "json" if file_name == "log_file" 
                        else "image" if file_name in image_map 
                        else "log"
                    )
                    if server_checksums.get(file_name) == checksum:
                        log_synced_file(file_name, checksum, "success", file_type)
                        print(f"{file_name} - Upload and verification successful.")
                    else:
                        log_synced_file(file_name, checksum, "failure", file_type)
                        print(f"{file_name} - Checksum mismatch or failure.")
                return True
            else:
                print("Server reported failure.")
                return False
        except json.JSONDecodeError:
            print("Failed to parse server response.")
            return False
    else:
        print(f"Upload failed: {response.status_code} - {response.text}")
        return False


# Main Sync Function
def sync_data():
    last_inspection_id = get_last_synced_inspection()
    print(f"Last synced inspection ID: {last_inspection_id}")
    log_file, image_map, start_id, end_id = generate_json_logs(last_inspection_id)
    unsynced_logs=fetch_unsynced_log_files()
    print(f"Unsynced logs: {len(unsynced_logs)}")

    if log_file:
        if upload_to_cloud(log_file, image_map,unsynced_logs):
            update_sync_history(start_id, end_id)

if __name__ == "__main__":
    sync_data()
