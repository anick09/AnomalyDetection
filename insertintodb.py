import sqlite3
import base64

DB_NAME = "/data/inspection_system_new5.db"

def encode_password(password):
    return base64.b64encode(password.encode()).decode()

def insert_dummy_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Check if users already exist
    cursor.execute("SELECT COUNT(*) FROM Users;")
    user_count = cursor.fetchone()[0]

    if user_count > 0:
        print("Users already exist. Skipping user insertion.")
        conn.close()
        return

    # Dummy users
    users = [
        ("admin", encode_password("admin123"), "admin", 3),
        ("user1", encode_password("password1"), "user", 3),
        ("user2", encode_password("password2"), "user", 3)
    ]

    # Insert dummy users
    cursor.executemany("INSERT INTO Users (username, password, role, retries_left) VALUES (?, ?, ?, ?)", users)

    # Commit and close
    conn.commit()
    conn.close()
    print("Dummy users added successfully!")

if __name__ == "__main__":
    insert_dummy_users()
