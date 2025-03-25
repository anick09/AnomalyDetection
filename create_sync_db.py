import sqlite3

def create_sync_server_db():
    """Creates sync_server.db and required tables."""
    conn = sqlite3.connect("sync_server2.db")  # Creates/opens the database
    cursor = conn.cursor()

    # Enable Foreign Key Constraints
    cursor.execute("PRAGMA foreign_keys = ON;")

    # ✅ Sync History Table - Tracks sync requests and prevents duplicate syncs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS SyncHistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            start_inspection_id INTEGER NOT NULL,
            end_inspection_id INTEGER NOT NULL
        );
    ''')

    # ✅ Log File Table - Stores information about synced log files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LogFile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    # ✅ Image File Table - Stores synced image details
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ImageFile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    # ✅ Sync Acknowledgment Table - Stores acknowledgment status of synced files
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS InspectionAcknowledgment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER,
            file_name TEXT NOT NULL,
            file_type TEXT CHECK(file_type IN ('log', 'image')),
            client_acknowledged INTEGER DEFAULT 0,
            last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ✅ LogFileSync Table - Stores log files that have been synced
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LogFileSync (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            client_acknowledged INTEGER DEFAULT 0
        )
    """)


    conn.commit()
    conn.close()
    print("sync_server.db created with required tables.")

if __name__ == "__main__":
    create_sync_server_db()
