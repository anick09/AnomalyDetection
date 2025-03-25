import requests
import os
import json
import time

# Constants
BASE_URL = "http://localhost:8000"  # Adjust if your FastAPI app runs on a different port
DOWNLOAD_DIR = "downloads"

# Create download directory if it doesn't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(DOWNLOAD_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(DOWNLOAD_DIR, "images"), exist_ok=True)

def test_sync_and_download():
    print("🚀 Testing sync and download functionality")
    
    # Step 1: Trigger the sync process
    print("\n--- Step 1: Triggering sync ---")
    response = requests.get(f"{BASE_URL}/sync/")
    if response.status_code != 200:
        print(f"❌ Sync failed with status code: {response.status_code}")
        return
    
    sync_data = response.json()
    print(f"✅ Sync initiated: {sync_data['message']}")
    
    # If no new data to sync, exit early
    if "No new data to sync" in sync_data['message']:
        print("No new data available. Test complete.")
        return
    
    # Extract log file and image list info
    log_file = sync_data.get('log_file')
    image_list = sync_data.get('image_list', [])

    print(image_list)
    
    print(f"Log file: {log_file}")
    print(f"Number of images: {len(image_list)}")
    
    # Step 2: Get available log files for download
    print("\n--- Step 2: Downloading log files ---")
    # In a real scenario, we'd query for available logs, but for this test we'll use log_id=1
    log_id = 1  # Assuming at least one log file exists
    log_response = requests.get(f"{BASE_URL}/download_log/{log_id}")
    
    if log_response.status_code == 200:
        # Get filename from Content-Disposition header if available
        filename = os.path.basename(log_file) if log_file else f"log_{log_id}.json"
        log_path = os.path.join(DOWNLOAD_DIR, "logs", filename)
        
        with open(log_path, "wb") as f:
            f.write(log_response.content)
        
        print(f"✅ Downloaded log file: {log_path}")
        
        # Acknowledge download
        ack_response = requests.post(f"{BASE_URL}/acknowledge/", params={"file_name": filename})
        print(f"Acknowledgment: {ack_response.json()['message']}")
        
        # Read and display some log content
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                inspection_count = len(log_data)
                print(f"Log contains {inspection_count} inspection records")
                if inspection_count > 0:
                    print(f"First inspection ID: {log_data[0]['inspection_id']}")
        except Exception as e:
            print(f"Error reading log file: {e}")
    else:
        print(f"❌ Failed to download log with ID {log_id}: {log_response.status_code}")
    
    # Step 3: Download images
    print("\n--- Step 3: Downloading images ---")
    # In a real scenario, we'd query for available images, but for this test we'll use image_ids 1-3
    for image_id in range(len(image_list)):
        image_response = requests.get(f"{BASE_URL}/download_image/{image_id}")
        
        if image_response.status_code == 200:
            # Try to get filename from headers
            content_disposition = image_response.headers.get('Content-Disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
            else:
                filename = f"image_{image_id}.jpg"
                
            image_path = os.path.join(DOWNLOAD_DIR, "images", filename)
            
            with open(image_path, "wb") as f:
                f.write(image_response.content)
            
            print(f"✅ Downloaded image: {image_path}")
            
            # Acknowledge download
            ack_response = requests.post(f"{BASE_URL}/acknowledge/", params={"file_name": filename})
            print(f"Acknowledgment: {ack_response.json()['message']}")
        else:
            print(f"❌ Failed to download image with ID {image_id}: {image_response.status_code}")
    
    print("\n✅ Inspection sync test completeded successfully!")


# # 📥 1. Get the list of unacknowledged logs
def fetch_unacknowledged_logs():
    response = requests.get(f"{BASE_URL}/list_logs/")
    if response.status_code == 200:
        return response.json().get("logs", [])
    else:
        print(f"❌ No logs available. Server response: {response.json()}")
        return []
    
# 📥 2. Download a specific log file
def download_log(file_name):
    response = requests.get(f"{BASE_URL}/download_log/", params={"file_name": file_name}, stream=True)
    
    if response.status_code == 200:
        # Save the file locally
        file_path = os.path.join("downloaded_logs", file_name)
        os.makedirs("downloaded_logs", exist_ok=True)
        
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        
        print(f"✅ Downloaded: {file_name}")
        return file_path
    else:
        print(f"❌ Failed to download {file_name}. Response: {response.json()}")
        return None

# 📥 3. Acknowledge a log file after downloading
def acknowledge_log(file_name):
    response = requests.post(f"{BASE_URL}/acknowledge_log/", json={"file_name": file_name})
    
    if response.status_code == 200:
        print(f"✅ Acknowledged: {file_name}")
    else:
        print(f"❌ Failed to acknowledge {file_name}. Response: {response.json()}")


if __name__ == "__main__":
    test_sync_and_download()

    # Step 4: Fetch unacknowledged logs
    # print("\n--- Step 4: Fetching unacknowledged logs ---")
    unacknowledged_logs = fetch_unacknowledged_logs()
    if not unacknowledged_logs:
        print("✅ No unacknowledged logs found.")
    else:
    
        for log in unacknowledged_logs:
            file_name = log["file_name"]
        
            # Download the log file
            file_path = download_log(file_name)
        
            if file_path:
                # Acknowledge the downloaded file
                acknowledge_log(file_name)