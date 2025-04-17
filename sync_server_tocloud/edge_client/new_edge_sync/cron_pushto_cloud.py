# import os
# import json
# import datetime
# import hashlib
# import sqlite3
# import requests
# import time
# import logging

# FILES_LIST_DBPATH = "/data/sync_service.db"
# CLOUD_SERVER_URL = "http://10.47.149.34:9002/api/upload"
# LOGS_DIR_LOGGING = "/data/server_sync_logs"
# os.makedirs(LOGS_DIR_LOGGING, exist_ok=True)

# # Logger Setup
# log_file_path = os.path.join(LOGS_DIR_LOGGING, "file_sync.log")
# logging.basicConfig(
#     filename=log_file_path,
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
# )
# logger = logging.getLogger(__name__)

# def generate_checksum(file_path):
#     sha256_hash = hashlib.sha256()
#     logger.info(f"Generating checksum for: {file_path}")
#     with open(file_path, "rb") as f:
#         for byte_block in iter(lambda: f.read(4096), b""):
#             sha256_hash.update(byte_block)
#     return sha256_hash.hexdigest()

# def log_synced_file(file_path, checksum, status):
#     conn = sqlite3.connect(FILES_LIST_DBPATH)
#     cursor = conn.cursor()
#     cursor.execute(
#         "UPDATE file_sync_table SET sync_status = ? WHERE file_path = ?",
#         (status, file_path),
#     )
#     conn.commit()
#     conn.close()
#     logger.info(f"Updated sync status for {file_path} to {status}.")


# def upload_to_cloud():
#     # Initialize database connection
#     try:
#         conn = sqlite3.connect(FILES_LIST_DBPATH)
#         cursor = conn.cursor()
#         cursor.execute("SELECT file_path FROM file_sync_table WHERE sync_status = 'pending'")
#         pending_files = cursor.fetchall()
#     except sqlite3.Error as e:
#         logger.error(f"Database error: {str(e)}")
#         return
#     finally:
#         if 'conn' in locals():
#             conn.close()

#     if not pending_files:
#         logger.info("No pending files to upload.")
#         return

#     checksums = {}
#     files = []

#     # Process pending files
#     for (file_path,) in pending_files:
#         if os.path.exists(file_path):
#             try:
#                 logger.info(f"Uploading: {file_path}")
#                 file_checksum = generate_checksum(file_path)
#                 checksums[file_path] = file_checksum
#                 files.append(("files", (os.path.basename(file_path), open(file_path, "rb"))))
#             except (IOError, OSError) as e:
#                 logger.error(f"Failed to process file {file_path}: {str(e)}")
#                 log_synced_file(file_path, "", "pending")
#         else:
#             logger.warning(f"File not found: {file_path}")
#             log_synced_file(file_path, "", "pending")

#     if not files:
#         logger.info("No valid files to upload.")
#         return

#     # Upload files to cloud
#     try:
#         response = requests.post(
#             CLOUD_SERVER_URL,
#             files=files,
#             data={"checksums": json.dumps(checksums)}
#         )
#     except requests.exceptions.ConnectionError:
#         logger.error(f"Failed to connect to server: {CLOUD_SERVER_URL}")
#         for file_path, _ in checksums.items():
#             log_synced_file(file_path, "", "pending")
#         return
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Upload failed due to: {str(e)}")
#         for file_path, _ in checksums.items():
#             log_synced_file(file_path, "", "pending")
#         return
#     finally:
#         # Close all file handles
#         for _, file_obj in files:
#             file_obj[1].close()

#     # Process server response
#     try:
#         response_data = response.json()
#         server_checksums = response_data.get("checksums", {})
#     except json.JSONDecodeError:
#         logger.error("Failed to parse server response.")
#         server_checksums = {}

#     # Verify checksums
#     for file_path, checksum in checksums.items():
#         filename = os.path.basename(file_path)

#         if filename not in server_checksums:
#             log_synced_file(file_path, checksum, "pending")
#             logger.warning(f"{file_path} - Not present in server response. Marked as pending.")
#             continue

#         server_checksum = server_checksums[filename]

#         if server_checksum == checksum:
#             log_synced_file(file_path, checksum, "success")
#             logger.info(f"{file_path} - Upload and verification successful.")
#         else:
#             log_synced_file(file_path, checksum, "pending")
#             logger.warning(f"{file_path} - Checksum mismatch. Marked as pending.")

#     if response.status_code != 200:
#         logger.error(f"Upload returned HTTP error: {response.status_code} - {response.text}")



# def main():
#     while True:
#         upload_to_cloud()
#         logger.info("Waiting for 6 minutes before next sync...")
#         time.sleep(360) # sleep for 6 minutes
    
# if __name__ == "__main__":
#     main()


import os
import json
import hashlib
import sqlite3
import requests
import time
import logging
from datetime import datetime
import pytz

# Constants
FILES_LIST_DBPATH = "/data/sync_service.db"
CLOUD_SERVER_URL = "http://192.168.1.74:9002/api/upload"
LOGS_DIR_LOGGING = "/data/server_sync_logs"
os.makedirs(LOGS_DIR_LOGGING, exist_ok=True)

# Timezone setup
SWISS_TZ = pytz.timezone("Europe/Zurich")

def get_swiss_time():
    return datetime.now(SWISS_TZ).strftime('%Y-%m-%d %H:%M:%S')

# Logger Setup
log_file_path = os.path.join(LOGS_DIR_LOGGING, "file_sync.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_checksum(file_path):
    sha256_hash = hashlib.sha256()
    logger.info(f"Generating checksum for: {file_path}")
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def log_synced_file(file_path, checksum, status):
    conn = sqlite3.connect(FILES_LIST_DBPATH)
    cursor = conn.cursor()
    swiss_time = get_swiss_time()
    cursor.execute(
        "UPDATE file_sync_table SET sync_status = ?, timestamp = ? WHERE file_path = ?",
        (status, swiss_time, file_path),
    )
    conn.commit()
    conn.close()
    logger.info(f"[{swiss_time}] Updated sync status for {file_path} to {status}.")

def upload_to_cloud():
    try:
        conn = sqlite3.connect(FILES_LIST_DBPATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM file_sync_table WHERE sync_status = 'pending'")
        pending_files = cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        return
    finally:
        if 'conn' in locals():
            conn.close()

    if not pending_files:
        logger.info("No pending files to upload.")
        return

    checksums = {}
    files = []

    for (file_path,) in pending_files:
        if os.path.exists(file_path):
            try:
                logger.info(f"Uploading: {file_path}")
                file_checksum = generate_checksum(file_path)
                checksums[file_path] = file_checksum
                files.append(("files", (os.path.basename(file_path), open(file_path, "rb"))))
            except (IOError, OSError) as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")
                log_synced_file(file_path, "", "pending")
        else:
            logger.warning(f"File not found: {file_path}")
            log_synced_file(file_path, "", "pending")

    if not files:
        logger.info("No valid files to upload.")
        return

    try:
        response = requests.post(
            CLOUD_SERVER_URL,
            files=files,
            data={"checksums": json.dumps(checksums)}
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to server: {CLOUD_SERVER_URL}")
        for file_path in checksums.keys():
            log_synced_file(file_path, "", "pending")
        return
    except requests.exceptions.RequestException as e:
        logger.error(f"Upload failed due to: {str(e)}")
        for file_path in checksums.keys():
            log_synced_file(file_path, "", "pending")
        return
    finally:
        for _, file_obj in files:
            file_obj[1].close()

    try:
        response_data = response.json()
        server_checksums = response_data.get("checksums", {})
    except json.JSONDecodeError:
        logger.error("Failed to parse server response.")
        server_checksums = {}

    for file_path, checksum in checksums.items():
        filename = os.path.basename(file_path)
        if filename not in server_checksums:
            log_synced_file(file_path, checksum, "pending")
            logger.warning(f"{file_path} - Not present in server response. Marked as pending.")
            continue

        server_checksum = server_checksums[filename]
        if server_checksum == checksum:
            log_synced_file(file_path, checksum, "success")
            logger.info(f"{file_path} - Upload and verification successful.")
        else:
            log_synced_file(file_path, checksum, "pending")
            logger.warning(f"{file_path} - Checksum mismatch. Marked as pending.")

    if response.status_code != 200:
        logger.error(f"Upload returned HTTP error: {response.status_code} - {response.text}")

def main():
    while True:
        logger.info(f"[{get_swiss_time()}] Starting sync process...")
        upload_to_cloud()
        logger.info(f"[{get_swiss_time()}] Waiting for 6 minutes before next sync...")
        time.sleep(360)  # sleep for 6 minutes

if __name__ == "__main__":
    main()
