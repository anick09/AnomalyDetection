import sqlite3

def create_tables():
    conn = sqlite3.connect("inspection_system_new2.db")  # Creates/opens a database file
    cursor = conn.cursor()
    
    # Enable Foreign Key Constraints
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            retries_left INTEGER DEFAULT 3,
            locked_until TIMESTAMP DEFAULT NULL
        );
    ''')
    
    # Session Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Session (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name VARCHAR(255) NOT NULL,
            user_id INTEGER NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP NULL,
            cam1_final_status INTEGER,
            cam2_final_status INTEGER,
            comment TEXT,
            FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE SET NULL
        );
    ''')
    
    # Inspection Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Inspection (
            inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            inspection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path1 TEXT,
            image_path2 TEXT,
            FOREIGN KEY (session_id) REFERENCES Session(session_id) ON DELETE CASCADE
        );
    ''')
    
    # BoundingBox_AI Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BoundingBox_AI (
            bbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            camera_index TINYINT CHECK (camera_index IN (0, 1)),
            x_min FLOAT NOT NULL,
            y_min FLOAT NOT NULL,
            x_max FLOAT NOT NULL,
            y_max FLOAT NOT NULL,
            confidence FLOAT NOT NULL,
            FOREIGN KEY (inspection_id) REFERENCES Inspection(inspection_id) ON DELETE CASCADE
        );
    ''')
    
    # Reviewer_Feedback Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Reviewer_Feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            camera_index TINYINT ,
            reviewer_id INTEGER NOT NULL,
            reviewer_comment TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (inspection_id) REFERENCES Inspection(inspection_id) ON DELETE CASCADE,
            FOREIGN KEY (reviewer_id) REFERENCES Users(id) ON DELETE SET NULL
        );
    ''')
    
    # BoundingBox_Missed Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BoundingBox_Missed (
            bbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL,
            x_min FLOAT NOT NULL,
            y_min FLOAT NOT NULL,
            x_max FLOAT NOT NULL,
            y_max FLOAT NOT NULL,
            FOREIGN KEY (feedback_id) REFERENCES Reviewer_Feedback(feedback_id) ON DELETE CASCADE
        );
    ''')
    
    conn.commit()
    conn.close()
    print("Tables created successfully.")

if __name__ == "__main__":
    create_tables()
