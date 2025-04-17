from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import os
import hashlib
import datetime
import sqlite3
import base64
import uvicorn
import logging
import json
from typing import List

app = FastAPI()

# Directory Setup
BASE_DIR = "data/uploads"
JSON_DIR = os.path.join(BASE_DIR, "json_logs")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LOG_DIR = os.path.join(BASE_DIR, "log_files")
DB_PATH = "data/file_uploads.db"

# Logger Setup with Named Logger
logger = logging.getLogger("FileSyncServer")  # Named logger
logger.setLevel(logging.INFO)

# Add file handler for logging
log_file_path = os.path.join(BASE_DIR, "server.log")
os.makedirs(BASE_DIR, exist_ok=True)  # Ensure base directory exists
if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def setup_directories():
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.info("Directories set up successfully.")

def setup_database():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS FileUploadHistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                checksum TEXT,
                upload_timestamp TEXT,
                status TEXT,
                file_type TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized.")

# Checksum Generation
def generate_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Log to Database
def log_file_upload(file_name, checksum, status, file_type):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.utcnow().isoformat()
    cursor.execute('''
        INSERT INTO FileUploadHistory (file_name, checksum, upload_timestamp, status, file_type)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_name, checksum, timestamp, status, file_type))
    conn.commit()
    conn.close()
    logger.info(f"Logged upload: {file_name} with status {status}")


@app.post("/api/upload")
async def upload_files(checksums: str = Form(...), files: list[UploadFile] = File(...)):
    logger.info(f"Received files: {[f.filename for f in files]}")
    logger.info(f"Received checksums: {checksums}")
    
    try:
        checksums_dict = json.loads(checksums)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse checksums: {e}")
        return {"status": "error", "message": f"Invalid checksums: {e}"}
    
    response_checksums = {}
    all_success = True
    
    for file in files:
        file_type = "json" if file.filename.endswith(".json") else "image" if file.filename.endswith((".jpg", ".png")) else "log"
        save_dir = JSON_DIR if file_type == "json" else IMAGE_DIR if file_type == "image" else LOG_DIR
        file_path = os.path.join(save_dir, file.filename)
        
        logger.info(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        checksum = generate_checksum(file_path)
        
        # Match filename to full path in checksums_dict
        client_checksum = None
        for key in checksums_dict:
            if os.path.basename(key) == file.filename:
                client_checksum = checksums_dict[key]
                logger.info(f"Matched {file.filename} to full path {key} with checksum {client_checksum}")
                break
        
        if client_checksum is None:
            logger.warning(f"No matching checksum found for {file.filename} in {checksums_dict}")
            status = "failure"
            all_success = False
        else:
            status = "success" if client_checksum == checksum else "failure"
            if status == "failure":
                all_success = False
            logger.info(f"Server checksum for {file_path}: {checksum}, Client checksum: {client_checksum}")
        
        response_checksums[file.filename] = checksum
        log_file_upload(file.filename, checksum, status, file_type)
    
    logger.info("Upload processing completed.")
    return {"status": "success" if all_success else "failure", "checksums": response_checksums}

# NEW: Endpoint to retrieve base64-encoded images
class ImageRequest(BaseModel):
    filename: str


@app.post("/api/get_image")
async def get_image(request: ImageRequest):
    filename = request.filename
    file_path = os.path.join(IMAGE_DIR, filename)
    
    logger.info(f"Received request for image: {filename}")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Image not found: {file_path}")
            # log_file_request(filename, None, "failure", "image", "retrieval")
            raise HTTPException(status_code=404, detail="Image not found")
        
        with open(file_path, "rb") as img_file:
            image_data = img_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        checksum = generate_checksum(file_path)
        # log_file_request(filename, checksum, "success", "image", "retrieval")
        logger.info(f"Successfully retrieved image: {filename}")
        
        return {"status": "success", "image_base64": image_base64}
    
    except Exception as e:
        logger.error(f"Failed to retrieve image {filename}: {e}")
        # log_file_request(filename, None, "failure", "image", "retrieval")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    setup_directories()
    setup_database()
    logger.info("Starting server on 0.0.0.0:9002")
    uvicorn.run("sync_server:app", host="0.0.0.0", port=9002, reload=True)

if __name__ == "__main__":
    main()