import sqlite3
import base64

# Connect to the database
conn = sqlite3.connect("inspection_system_new2.db")
cursor = conn.cursor()

# Function to encode password
def encode_password(password):
    return base64.b64encode(password.encode()).decode()

# Dummy users
users = [
    ("admin", encode_password("admin123"), "admin", 3),
    ("user1", encode_password("password1"), "user", 3),
    ("user2", encode_password("password2"), "user", 3)
]

# Insert dummy users
cursor.executemany("INSERT OR IGNORE INTO users (username, password, role, retries_left) VALUES (?, ?, ?, ?)", users)

# Commit and close
conn.commit()
conn.close()

print("Dummy users added successfully!")
