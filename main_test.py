import pytest
import sqlite3
import base64
from fastapi.testclient import TestClient
from main import app, init_db  # Import FastAPI app and DB initializer

# Initialize the test client
client = TestClient(app)

# Helper function to encode passwords
def encode_password(password):
    return base64.b64encode(password.encode()).decode()

# Setup Test Database
def setup_test_db():
    conn = sqlite3.connect("inspection_system_test.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS users")  # Reset table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            retries_left INTEGER DEFAULT 3
        )
    """)
    cursor.executemany(
        "INSERT INTO users (username, password, role, retries_left) VALUES (?, ?, ?, ?)",
        [
            ("admin", encode_password("admin123"), "admin", 3),
            ("user1", encode_password("password1"), "user", 3),
        ]
    )
    conn.commit()
    conn.close()

# Run before tests
@pytest.fixture(scope="module", autouse=True)
def setup():
    setup_test_db()

# Test Cases

def test_login_success():
    """Test login with correct credentials"""
    response = client.post("/login", json={"UserName": "admin", "Password": encode_password("admin123")})
    assert response.status_code == 200
    assert response.json()["status"] == "Success"
    assert response.json()["Username"] == "admin"

def test_login_wrong_password():
    """Test login with incorrect password"""
    response = client.post("/login", json={"UserName": "admin", "Password": encode_password("wrongpass")})
    assert response.status_code == 401
    assert "Invalid password" in response.json()["detail"]

def test_login_wrong_username():
    """Test login with non-existent user"""
    response = client.post("/login", json={"UserName": "unknown", "Password": encode_password("admin123")})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"

def test_login_account_lock():
    """Test account lockout after 3 failed attempts"""
    for _ in range(3):
        response = client.post("/login", json={"UserName": "admin", "Password": encode_password("wrongpass")})
    
    response = client.post("/login", json={"UserName": "admin", "Password": encode_password("wrongpass")})
    assert response.status_code == 403
    assert response.json()["detail"] == "Account locked"

def test_login_missing_fields():
    """Test login with missing fields"""
    response = client.post("/login", json={"UserName": "admin"})
    assert response.status_code == 422  # Unprocessable Entity

