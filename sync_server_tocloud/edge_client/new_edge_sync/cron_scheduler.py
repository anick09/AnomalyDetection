import sqlite3
from datetime import datetime
import time
import schedule
import os

# Initialize the database connection
# os.makedirs("data", exist_ok=True)
conn = sqlite3.connect('/data/sync_service.db')
cursor = conn.cursor()

# Create the cron_table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS cron_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
)
''')
conn.commit()

def populate_cron_table():
    """Inserts a new row with the current timestamp and default status."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
    INSERT INTO cron_table (timestamp, status) VALUES (?, 'pending')
    ''', (timestamp,))
    conn.commit()
    print(f"Inserted row with timestamp: {timestamp} and status: 'pending'")

if __name__ == "__main__":
    try:
        print("Starting daily cron service...")
        # Schedule the cron job at 1 AM daily
        # schedule.every().day.at("15:17").do(populate_cron_table)

        #schedule at every 1200 seconds for testing
        schedule.every(1200).seconds.do(populate_cron_table)

        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("Cron service stopped.")
    finally:
        conn.close()
