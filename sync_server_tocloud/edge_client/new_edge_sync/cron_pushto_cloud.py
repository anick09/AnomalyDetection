import os
import json
import datetime
import hashlib
import sqlite3
import requests
import time
import logging

FILES_LIST_DBPATH = "/data/sync_service.db"
CLOUD_SERVER_URL = "http://192.168.1.74:9002/api/upload"
LOGS_DIR_LOGGING = "/data/server_sync_logs"
os.makedirs(LOGS_DIR_LOGGING, exist_ok=True)

# Logger Setup
log_file_path = os.path.join(LOGS_DIR_LOGGING, "file_sync.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
    cursor.execute(
        "UPDATE file_sync_table SET sync_status = ? WHERE file_path = ?",
        (status, file_path),
    )
    conn.commit()
    conn.close()
    logger.info(f"Updated sync status for {file_path} to {status}.")

def upload_to_cloud():
    conn = sqlite3.connect(FILES_LIST_DBPATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT file_path FROM file_sync_table WHERE sync_status = 'pending'")
    pending_files = cursor.fetchall()
    conn.close()
    
    if not pending_files:
        logger.info("No pending files to upload.")
        return
    
    checksums = {}
    files = []
    
    for (file_path,) in pending_files:
        if os.path.exists(file_path):
            logger.info(f"Uploading: {file_path}")
            file_checksum = generate_checksum(file_path)
            checksums[file_path] = file_checksum
            files.append(("files", (os.path.basename(file_path), open(file_path, "rb"))))
        else:
            logger.warning(f"File not found: {file_path}")
            log_synced_file(file_path, "", "failed")
    
    if not files:
        logger.info("No valid files to upload.")
        return
    
    response = requests.post(
        CLOUD_SERVER_URL,
        files=files,
        data={"checksums": json.dumps(checksums)}
    )
    
    for _, file_obj in files:
        file_obj[1].close()
    
    if response.status_code == 200:
        try:
            response_data = response.json()
            if response_data.get("status") == "success":
                server_checksums = response_data.get("checksums", {})
                for file_path, checksum in checksums.items():
                    filename = os.path.basename(file_path)
                    if server_checksums.get(filename) == checksum:
                        log_synced_file(file_path, checksum, "success")
                        logger.info(f"{file_path} - Upload and verification successful.")
                    else:
                        log_synced_file(file_path, checksum, "pending")
                        logger.warning(f"{file_path} - Checksum mismatch or failure.")
            else:
                logger.error("Server reported failure.")
        except json.JSONDecodeError:
            logger.error("Failed to parse server response.")
    else:
        logger.error(f"Upload failed: {response.status_code} - {response.text}")

def main():
    while True:
        upload_to_cloud()
        logger.info("Waiting for 2 minutes before next sync...")
        time.sleep(360) # sleep for 6 minutes
    
if __name__ == "__main__":
    main()
