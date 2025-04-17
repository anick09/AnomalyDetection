# import sqlite3
# import time
# import schedule #type: ignore
# import logging
# from datetime import datetime
# import os
# import json
# import hashlib
# import requests
# import threading
# import signal
# from datetime import datetime
# import pytz

# # Directories
# INSPECTION_DB_PATH = "/data/inspection_system_new5.db"
# LOGS_DIR = "/data/jsonlogs"
# LOGS_DIR1 = "/data/main_logs"
# LOGS_DIR2 = "/data/process_image_logs"
# LOGS_DIR_LOGGING = "/data/server_sync_logs"
# os.makedirs(LOGS_DIR, exist_ok=True)
# os.makedirs(LOGS_DIR_LOGGING, exist_ok=True)

# # Logger Setup
# log_file_path = os.path.join(LOGS_DIR_LOGGING, "process_cron.log")
# logging.basicConfig(
#     filename=log_file_path,
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
# )
# logger = logging.getLogger(__name__)

# # Initialize the database connection
# conn = sqlite3.connect('/data/sync_service.db', timeout=10)
# cursor = conn.cursor()
# conn_inspection = sqlite3.connect(INSPECTION_DB_PATH, timeout=10)
# cursor_inspection = conn_inspection.cursor()

# # Ensure file_sync_table exists
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS file_sync_table (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     file_path TEXT NOT NULL,
#     file_type TEXT NOT NULL,
#     timestamp TEXT NOT NULL,
#     sync_status TEXT NOT NULL DEFAULT 'pending'
# )
# ''')

# cursor.execute('''
# CREATE TABLE IF NOT EXISTS SyncHistory (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     start_inspection_id INTEGER NOT NULL,
#     end_inspection_id INTEGER NOT NULL,
#     sync_timestamp TEXT NOT NULL
# )
# ''')
# conn.commit()


# # Job lock
# job_lock = threading.Lock()

# # Functions (unchanged except for connection reuse)
# def get_pending_cron_entry():
#     global cursor
#     cursor.execute("SELECT id, timestamp FROM cron_table WHERE status = 'pending' ORDER BY timestamp ASC LIMIT 1;")
#     return cursor.fetchone()

# def update_cron_status(cron_id, status):
#     global cursor, conn
#     cursor.execute("UPDATE cron_table SET status = ? WHERE id = ?", (status, cron_id))
#     conn.commit()


# def get_last_synced_inspection():
#     cursor.execute("SELECT end_inspection_id FROM SyncHistory ORDER BY id DESC LIMIT 1")
#     last_sync = cursor.fetchone()
#     return last_sync[0] if last_sync else 0

# def update_sync_history(start_id, end_id):
#     cursor.execute("""
#         INSERT INTO SyncHistory (start_inspection_id, end_inspection_id, sync_timestamp)
#         VALUES (?, ?, ?)
#     """, (start_id, end_id, datetime.utcnow().isoformat()))
#     conn.commit()

# def fetch_new_inspections(last_inspection_id,time_stamp):
#     current_date = time_stamp
#     query = """
#         SELECT inspection_id, session_id, inspection_timestamp, image_path1, image_path2 
#         FROM Inspection
#         WHERE inspection_id > ? AND DATE(inspection_timestamp) < ?
#     """
#     cursor_inspection.execute(query, (last_inspection_id, current_date))
#     return cursor_inspection.fetchall()

# # Fetch Inspection Details
# def fetch_inspection_details(inspection_id):
#     conn = None
#     try:
#         conn = sqlite3.connect(INSPECTION_DB_PATH)
#         cursor = conn.cursor()

#         cursor.execute("""
#             SELECT camera_index, x_min, y_min, x_max, y_max, confidence
#             FROM BoundingBox_AI WHERE inspection_id = ?
#         """, (inspection_id,))
#         bounding_boxes = cursor.fetchall()

#         cursor.execute("""
#             SELECT camera_index, reviewer_comment, Anomalies_found, Anomalies_missed
#             FROM Reviewer_Feedback WHERE inspection_id = ?
#         """, (inspection_id,))
#         feedback = cursor.fetchall()

#         cursor.execute("""
#             SELECT camera_index, x_min, y_min, width, height, type, comment
#             FROM False_Annotations WHERE inspection_id = ?
#         """, (inspection_id,))
#         false_annotations = cursor.fetchall()

#         return bounding_boxes, feedback, false_annotations
#     finally:
#         if conn:
#             conn.close()


# # Generate JSON Logs
# def generate_json_logs(last_inspection_id,timestamp):
#     if last_inspection_id is None:
#         last_inspection_id = 0
#     new_inspections = fetch_new_inspections(last_inspection_id,timestamp)
#     print(len(new_inspections))

#     if not new_inspections:
#         return None, {}, None, None

#     log_data = []
#     image_map = {}

#     start_id = new_inspections[0][0]
#     end_id = new_inspections[-1][0]

#     for inspection in new_inspections:
#         inspection_id, session_id, timestamp, img1, img2 = inspection
#         bounding_boxes, feedback, false_annotations = fetch_inspection_details(inspection_id)

#         log_entry = {
#             "inspection_id": inspection_id,
#             "session_id": session_id,
#             "timestamp": timestamp,
#             "images": {"image1": img1, "image2": img2},
#             "bounding_boxes": [
#                 {"camera": cam, "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "confidence": conf}
#                 for cam, x1, y1, x2, y2, conf in bounding_boxes
#             ],
#             "feedback": [
#                 {"camera": cam, "comment": comment, "found": found, "missed": missed}
#                 for cam, comment, found, missed in feedback
#             ],
#             "false_annotations": [
#                 {"camera": cam, "x_min": x1, "y_min": y1, "width": w, "height": h, "type": t, "comment": c}
#                 for cam, x1, y1, w, h, t, c in false_annotations
#             ]
#         }
#         log_data.append(log_entry)
#         if img1:
#             image_map[img1] = inspection_id
#         if img2:
#             image_map[img2] = inspection_id

#     log_file = os.path.join(LOGS_DIR, f"logs_{int(datetime.utcnow().timestamp())}.json")
#     with open(log_file, "w") as f:
#         json.dump(log_data, f, indent=4)

#     return log_file, image_map, start_id, end_id

# # Generate Checksum
# def generate_checksum(file_path):
#     sha256_hash = hashlib.sha256()
#     print(f"Generating checksum for: {file_path}")
#     with open(file_path, "rb") as f:
#         for byte_block in iter(lambda: f.read(4096), b""):
#             sha256_hash.update(byte_block)
#     return sha256_hash.hexdigest()


# def populate_file_sync_table(timestamp):
#     """Populates the file_sync_table with pending files to sync."""

#     json_log_file, image_map, start_id, end_id = generate_json_logs(get_last_synced_inspection(),timestamp)
#     json_files = [json_log_file]
#     image_files = list(image_map.keys())
#     log_files = []

#     # Use UTC to match log naming/modification timezone (adjust if needed)
#     now_utc = datetime.now(pytz.UTC)
#     today_str = now_utc.strftime('%Y-%m-%d')  # To compare with log file date

#     for dir_path in [LOGS_DIR1, LOGS_DIR2]:
#         for root, _, files in os.walk(dir_path):
#             for file in files:
#                 if file.endswith(".log"):
#                     file_path = os.path.join(root, file)
#                     file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=pytz.UTC)
#                     file_date_str = file_mtime.strftime('%Y-%m-%d')

#                     # Skip if file is from today
#                     if file_date_str >= today_str:
#                         continue

#                     # Also check against provided timestamp if needed
#                     if file_mtime < datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC):
#                         # Check if already synced
#                         cursor.execute("SELECT COUNT(*) FROM file_sync_table WHERE file_path = ?", (file_path,))
#                         count = cursor.fetchone()[0]
#                         if count == 0:
#                             log_files.append(file_path)

#         print(f"Number of log files: {len(log_files)}")

    

#     if(json_log_file):
#         #update sync history
#         update_sync_history(start_id, end_id)

#         for file in json_files:
#             cursor.execute('''
#             INSERT INTO file_sync_table (file_path, file_type, timestamp, sync_status)
#             VALUES (?, 'json', ?, 'pending')
#             ''', (file, timestamp))

#         for file in image_files:
#             cursor.execute('''
#             INSERT INTO file_sync_table (file_path, file_type, timestamp, sync_status)
#             VALUES (?, 'image', ?, 'pending')
#             ''', (file, timestamp))

#     if(log_files):
#         for file in log_files:
#             cursor.execute('''
#             INSERT INTO file_sync_table (file_path, file_type, timestamp, sync_status)
#             VALUES (?, 'log', ?, 'pending')
#             ''', (file, timestamp))

#         conn.commit()
#         print(f"Populated file_sync_table with files from {timestamp}")
#     else:
#         print(f"No new files to populate for timestamp: {timestamp}")

# def run_cron_job():
#     if not job_lock.acquire(blocking=False):
#         logger.info("Another job is already running, skipping this run.")
#         return
#     try:
#         cron_entry = get_pending_cron_entry()
#         if cron_entry:
#             cron_id, timestamp = cron_entry
#             logger.info(f"Found pending cron job: ID {cron_id}, Timestamp {timestamp}")
#             populate_file_sync_table(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#             update_cron_status(cron_id, "complete")
#             logger.info(f"Cron job {cron_id} completed successfully.")
#     except Exception as e:
#         logger.error(f"Error processing cron job {cron_id}: {e}")
#     finally:
#         job_lock.release()

# # Cleanup on exit
# def cleanup(signum, frame):
#     conn.close()
#     conn_inspection.close()
#     logger.info("Database connections closed.")
#     exit(0)

# signal.signal(signal.SIGINT, cleanup)
# signal.signal(signal.SIGTERM, cleanup)

# # Schedule the job
# schedule.every(8).minutes.do(run_cron_job)

# # Keep running the scheduler
# while True:
#     schedule.run_pending()
#     time.sleep(10)


import sqlite3
import time
import schedule  # type: ignore
import logging
from datetime import datetime
import os
import json
import hashlib
import requests
import threading
import signal
import pytz

# Timezone helper
def get_swiss_time():
    return datetime.now(pytz.timezone("Europe/Zurich"))

def to_swiss(dt: datetime) -> datetime:
    tz = pytz.timezone("Europe/Zurich")
    return tz.localize(dt) if dt.tzinfo is None else dt.astimezone(tz)

# Directories
INSPECTION_DB_PATH = "/data/inspection_system_new5.db"
LOGS_DIR = "/data/jsonlogs"
LOGS_DIR1 = "/data/main_logs"
LOGS_DIR2 = "/data/process_image_logs"
LOGS_DIR_LOGGING = "/data/server_sync_logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR_LOGGING, exist_ok=True)

# Logger Setup
log_file_path = os.path.join(LOGS_DIR_LOGGING, "process_cron.log")
class SwissFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, pytz.timezone("Europe/Zurich"))
        return dt.timetuple()

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        return time.strftime(datefmt or "%Y-%m-%d %H:%M:%S", ct)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(SwissFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(SwissFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Initialize the database connection
conn = sqlite3.connect('/data/sync_service.db', timeout=10)
cursor = conn.cursor()
conn_inspection = sqlite3.connect(INSPECTION_DB_PATH, timeout=10)
cursor_inspection = conn_inspection.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS file_sync_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    sync_status TEXT NOT NULL DEFAULT 'pending'
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS SyncHistory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_inspection_id INTEGER NOT NULL,
    end_inspection_id INTEGER NOT NULL,
    sync_timestamp TEXT NOT NULL
)
''')
conn.commit()

job_lock = threading.Lock()

def get_pending_cron_entry():
    cursor.execute("SELECT id, timestamp FROM cron_table WHERE status = 'pending' ORDER BY timestamp ASC LIMIT 1;")
    return cursor.fetchone()

def update_cron_status(cron_id, status):
    cursor.execute("UPDATE cron_table SET status = ? WHERE id = ?", (status, cron_id))
    conn.commit()

def get_last_synced_inspection():
    cursor.execute("SELECT end_inspection_id FROM SyncHistory ORDER BY id DESC LIMIT 1")
    last_sync = cursor.fetchone()
    return last_sync[0] if last_sync else 0

def update_sync_history(start_id, end_id):
    cursor.execute("""
        INSERT INTO SyncHistory (start_inspection_id, end_inspection_id, sync_timestamp)
        VALUES (?, ?, ?)
    """, (start_id, end_id, get_swiss_time().isoformat()))
    conn.commit()

def fetch_new_inspections(last_inspection_id, time_stamp):
    query = """
        SELECT inspection_id, session_id, inspection_timestamp, image_path1, image_path2 
        FROM Inspection
        WHERE inspection_id > ? AND DATE(inspection_timestamp) < ?
    """
    cursor_inspection.execute(query, (last_inspection_id, time_stamp))
    return cursor_inspection.fetchall()

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

def generate_json_logs(last_inspection_id, timestamp):
    if last_inspection_id is None:
        last_inspection_id = 0
    new_inspections = fetch_new_inspections(last_inspection_id, timestamp)

    if not new_inspections:
        return None, {}, None, None

    log_data = []
    image_map = {}

    start_id = new_inspections[0][0]
    end_id = new_inspections[-1][0]

    for inspection in new_inspections:
        inspection_id, session_id, ts, img1, img2 = inspection
        bounding_boxes, feedback, false_annotations = fetch_inspection_details(inspection_id)

        log_entry = {
            "inspection_id": inspection_id,
            "session_id": session_id,
            "timestamp": ts,
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

    log_file = os.path.join(LOGS_DIR, f"logs_{int(get_swiss_time().timestamp())}.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    return log_file, image_map, start_id, end_id

def generate_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def populate_file_sync_table(timestamp):
    json_log_file, image_map, start_id, end_id = generate_json_logs(get_last_synced_inspection(), timestamp)
    json_files = [json_log_file]
    image_files = list(image_map.keys())
    log_files = []

    now_swiss = get_swiss_time()
    today_str = now_swiss.strftime('%Y-%m-%d')

    for dir_path in [LOGS_DIR1, LOGS_DIR2]:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".log"):
                    file_path = os.path.join(root, file)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=pytz.UTC)
                    file_date_str = file_mtime.astimezone(pytz.timezone("Europe/Zurich")).strftime('%Y-%m-%d')

                    if file_date_str >= today_str:
                        continue

                    if file_mtime < to_swiss(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')):
                        cursor.execute("SELECT COUNT(*) FROM file_sync_table WHERE file_path = ?", (file_path,))
                        if cursor.fetchone()[0] == 0:
                            log_files.append(file_path)

    if json_log_file:
        update_sync_history(start_id, end_id)

        for file in json_files:
            cursor.execute('''
            INSERT INTO file_sync_table (file_path, file_type, timestamp, sync_status)
            VALUES (?, 'json', ?, 'pending')
            ''', (file, timestamp))

        for file in image_files:
            cursor.execute('''
            INSERT INTO file_sync_table (file_path, file_type, timestamp, sync_status)
            VALUES (?, 'image', ?, 'pending')
            ''', (file, timestamp))

    for file in log_files:
        cursor.execute('''
        INSERT INTO file_sync_table (file_path, file_type, timestamp, sync_status)
        VALUES (?, 'log', ?, 'pending')
        ''', (file, timestamp))

    conn.commit()
    logger.info(f"Populated file_sync_table with files from {timestamp}")

def run_cron_job():
    if not job_lock.acquire(blocking=False):
        logger.info("Another job is already running, skipping this run.")
        return
    try:
        cron_entry = get_pending_cron_entry()
        if cron_entry:
            cron_id, timestamp = cron_entry
            logger.info(f"Found pending cron job: ID {cron_id}, Timestamp {timestamp}")
            populate_file_sync_table(get_swiss_time().strftime('%Y-%m-%d %H:%M:%S'))
            update_cron_status(cron_id, "complete")
            logger.info(f"Cron job {cron_id} completed successfully.")
    except Exception as e:
        logger.error(f"Error processing cron job {cron_id if 'cron_id' in locals() else 'UNKNOWN'}: {e}")
    finally:
        job_lock.release()

def cleanup(signum, frame):
    conn.close()
    conn_inspection.close()
    logger.info("Database connections closed.")
    exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

schedule.every(8).minutes.do(run_cron_job)

while True:
    schedule.run_pending()
    time.sleep(10)
