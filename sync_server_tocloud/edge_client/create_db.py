import sqlite3

# Define the database path
db_path = "data/sync_server2.db"

# Connect to the database (creates it if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# SQL script from your canvas
sql_script = '''
-- Table: SyncHistory
CREATE TABLE IF NOT EXISTS SyncHistory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_inspection_id INTEGER,
    end_inspection_id INTEGER,
    sync_timestamp TEXT
);

-- Table: FileSyncHistory
CREATE TABLE IF NOT EXISTS FileSyncHistory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    checksum TEXT,
    sync_timestamp TEXT,
    status TEXT,
    file_type TEXT
);
'''

# Execute the SQL script
cursor.executescript(sql_script)
conn.commit()
conn.close()

print("Database and tables created successfully!")
