from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import os
import hashlib
import datetime
import sqlite3
import uvicorn

app = FastAPI()

# Directory Setup
BASE_DIR = "data/uploads"
JSON_DIR = os.path.join(BASE_DIR, "json_logs")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LOG_DIR = os.path.join(BASE_DIR, "log_files")
DB_PATH = "data/file_uploads.db"

def setup_directories():
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

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

# Endpoint to Receive Files
@app.post("/api/upload")
async def upload_files(checksums: str = Form(...), files: list[UploadFile] = File(...)):
    checksums = eval(checksums)  # Convert string to dict
    response_checksums = {}
    
    for file in files:
        file_type = "json" if file.filename.endswith(".json") else "image" if file.filename.endswith((".jpg", ".png")) else "log"
        save_dir = JSON_DIR if file_type == "json" else IMAGE_DIR if file_type == "image" else LOG_DIR
        file_path = os.path.join(save_dir, file.filename)
        
        # Save File
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Verify Checksum
        checksum = generate_checksum(file_path)
        status = "success" if checksums.get(file.filename) == checksum else "failure"
        response_checksums[file.filename] = checksum
        
        # Log in DB
        log_file_upload(file.filename, checksum, status, file_type)
    
    return {"status": "success", "checksums": response_checksums}

def main():
    setup_directories()
    setup_database()
    uvicorn.run("sync_server:app", host="0.0.0.0", port=8002, reload=True)

if __name__ == "__main__":
    main()
