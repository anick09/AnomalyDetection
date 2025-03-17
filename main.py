from fastapi import FastAPI, HTTPException, Depends, Response, WebSocket
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import sqlite3
import base64
# import datetime
from pathlib import Path
import cv2
import asyncio
import json,os
from datetime import datetime 
import pytz
from typing import List
from process_image import find_anomaly,check_camera_blockage,is_black_or_white_screen
import logging
from fastapi.middleware.cors import CORSMiddleware
import shutil
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


swiss_tz = pytz.timezone('Europe/Zurich')


LOG_DIR = "main_logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Unique log file for main.py
# log_filename = f'{LOG_DIR}/main_{datetime.now(swiss_tz).strftime("%Y%m%d")}.log'

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_filename),
#         logging.StreamHandler()
#     ]
# )

# logger_main = logging.getLogger("main")
# logger_main.info("Logging initialized in main_logs")

# Configure logging
# Unique log file for process_image.py
log_filename = f'{LOG_DIR}/process_image_{datetime.now(pytz.timezone("Europe/Zurich")).strftime("%Y%m%d")}.log'

# Configure a separate logger for process_image.py
logger_main = logging.getLogger("main")  # <== Named Logger
logger_main.setLevel(logging.INFO)

# Prevent duplicate handlers if script is re-imported
if not logger_main.hasHandlers():
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger_main.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger_main.addHandler(console_handler)

logger_main.info("Logging initialized in main_logs")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Request model
class LoginRequest(BaseModel):
    username: str
    password: str

# Response model
class LoginResponse(BaseModel):
    status: str
    Username: str
    User_id: int
    Role: str
    Timestamp: str
    RetriesLeft: int

# Database connection function
def get_db():
    try:
        conn = sqlite3.connect("inspection_system_new5.db")
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        logger_main.error(f"Database connection error: {str(e)}")
        raise
##------------------------------------------------------------------------------------------------------------------------------------------------------
# from datetime import datetime, timedelta

# import logging
# from datetime import datetime
# import os

# # Add at top of file after imports
# if not os.path.exists("logs"):
#     os.makedirs("logs")

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'logs/app_{datetime.now(swiss_tz).strftime("%Y%m%d")}.log'),
#         logging.StreamHandler()
#     ]
# )


# ...existing code...

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    logger_main.info(f"Login attempt for user: {request.username}")
    conn, cursor = get_db()

    try:
        cursor.execute("SELECT * FROM users WHERE username=?", (request.username,))
        user = cursor.fetchone()
        if not user:
            logger_main.warning(f"Failed login attempt - user not found: {request.username}")
            raise HTTPException(status_code=401, detail="Invalid username or password")

        columns = [col[0] for col in cursor.description]
        user = dict(zip(columns, user))

        swiss_tz = pytz.timezone('Europe/Zurich')  # Time zone for Switzerland
        swiss_time = datetime.now(swiss_tz)
        # now = datetime.utcnow()
        now = swiss_time

        # Check if account is locked
        if user["locked_until"] and now < datetime.fromisoformat(user["locked_until"]):
            remaining_time = (datetime.fromisoformat(user["locked_until"]) - now).seconds
            logger_main.warning(f"Login blocked - account locked: {request.username}")
            raise HTTPException(status_code=403, detail=f"Account locked. Try again in {remaining_time} seconds.")

        # Decode passwords
        decoded_password = base64.b64decode(request.password).decode()
        stored_password = base64.b64decode(user["password"]).decode()

        if decoded_password != stored_password:
            retries_left = user["retries_left"] - 1
            locked_until = None

            if retries_left <= 0:
                locked_until = (now + timedelta(minutes=30)).isoformat()
                retries_left = 0
                logger_main.warning(f"Account locked due to multiple failed attempts: {request.username}")

            cursor.execute("UPDATE users SET retries_left=?, locked_until=? WHERE username=?", 
                           (retries_left, locked_until, request.username))
            conn.commit()

            if locked_until:
                raise HTTPException(status_code=403, detail=f"Account locked for 30 minutes.")
            
            logger_main.warning(f"Failed login attempt for {request.username}. {retries_left} retries left")
            raise HTTPException(status_code=401, detail=f"Invalid password. {retries_left} retries left.")

        # Successful login - reset retries and unlock account
        cursor.execute("UPDATE users SET retries_left=3, locked_until=NULL WHERE username=?", (request.username,))
        conn.commit()
        
        logger_main.info(f"Successful login for user: {request.username}")
        return LoginResponse(
            status="Success",
            Username=request.username,
            User_id=user["id"],
            Role=user["role"],
            Timestamp=now.isoformat(),
            RetriesLeft=3
        )
    except Exception as e:
        logger_main.error(f"Login error for user {request.username}: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


async def process_images(image1, image2):
    #Read roi data from file
    roi_data = {}
    if os.path.exists("roi_data.json"):
        with open("roi_data.json", "r") as f:
            roi_data = json.load(f)
    
    # Crop images using ROI data
    roi1 = roi_data["0"]
    roi2 = roi_data["2"]

    cropped_image1 = image1[roi1["y"]:roi1["y"] + roi1["height"], roi1["x"]:roi1["x"] + roi1["width"]]
    cropped_image2 = image2[roi2["y"]:roi2["y"] + roi2["height"], roi2["x"]:roi2["x"] + roi2["width"]]

    #Temporarily save cropped images
    cv2.imwrite("cropped1.jpg", cropped_image1)
    cv2.imwrite("cropped2.jpg", cropped_image2)

    # Process images
    BoundingBoxes1=find_anomaly("roi_camera_0.jpg", "cropped1.jpg", 0 ,roi1,metric='cosine', threshold_method='fixed')
    BoundingBoxes2=find_anomaly("roi_camera_2.jpg", "cropped2.jpg", 2 ,roi2,metric='cosine', threshold_method='fixed')


    return BoundingBoxes1, BoundingBoxes2


class InspectRequest(BaseModel):
    session_id: int


@app.websocket("/inspect")
async def inspect(websocket: WebSocket):
    await websocket.accept()
    logger_main.info("WebSocket connection accepted")
    
    try:
        data = await websocket.receive_json()
        request = InspectRequest(**data)
        session_id = request.session_id
        logger_main.info(f"Received inspection request for session_id: {session_id}")

        conn, cursor = get_db()

        # Check if session exists
        cursor.execute("SELECT * FROM Session WHERE session_id = ?", (session_id,))
        session = cursor.fetchone()
        if session is None:
            logger_main.warning(f"Session not found for session_id: {session_id}")
            await websocket.send_json({"error": "Session not found"})
            await websocket.close()
            conn.close()
            return

        # Insert a new inspection record
        cursor.execute("INSERT INTO Inspection (session_id) VALUES (?)", (session_id,))
        conn.commit()
        inspection_id = cursor.lastrowid
        logger_main.info(f"Created new inspection record with id: {inspection_id}")
        

        #Read reference images from files
        images = sorted(
            [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")],  # Filter only .jpg images
            key=lambda x: x[0]  # Sort by the first character of the filename
        )
        #Reading reference image from files
        ref_img1= cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
        _, img_encoded = cv2.imencode(".jpg", ref_img1)
        ref_img1_encoded= img_encoded.tobytes()
        # log
        logger_main.info("Reference image 1 read and encoded")

        ref_img2= cv2.imread(os.path.join(IMAGE_FOLDER, images[1]))
        _, img_encoded = cv2.imencode(".jpg", ref_img2)
        ref_img2_encoded= img_encoded.tobytes()
        # log
        logger_main.info("Reference image 2 read and encoded")

        await websocket.send_json({
            "image1": base64.b64encode(ref_img1_encoded).decode("utf-8"),
            "image2": base64.b64encode(ref_img2_encoded).decode("utf-8"),
            "bounding_boxes1": [],
            "bounding_boxes2": []
        })
        #log 
        logger_main.info("Reference images sent to UI")



        # Capture images
        logger_main.info("Attempting to capture images from cameras")
        #time to capture and encode image
        image1 = capture_image(0)
        image2 = capture_image(2)

        if image1 is None and image2 is None:
            logger_main.error("Both Camera 0 and Camera 2 are disconnected.")
            await websocket.send_json({"error": "Both Camera 0 and Camera 2 are disconnected. Please check the connection."})
            await websocket.close()
            conn.close()
            return
        elif image1 is None:
            logger_main.error("Camera 0 is disconnected.")
            await websocket.send_json({"error": "Camera 0 is disconnected. Please check the connection."})
            await websocket.close()
            conn.close()
            return
        elif image2 is None:
            logger_main.error("Camera 2 is disconnected.")
            await websocket.send_json({"error": "Camera 2 is disconnected. Please check the connection."})
            await websocket.close()
            conn.close()
            return


        _, img_encoded = cv2.imencode(".jpg", image1)
        img1 = img_encoded.tobytes()


        _, img_encoded = cv2.imencode(".jpg", image2)
        img2 = img_encoded.tobytes()



        

        # ...existing code...
        # Create directory if not exists
        if not os.path.exists("full_image_inspection"):
            os.makedirs("full_image_inspection")
        
        cv2.imwrite(f"full_image_inspection/{inspection_id}_0.jpg", image1)
        cv2.imwrite(f"full_image_inspection/{inspection_id}_2.jpg", image2)
        
        image_path1 = f"full_image_inspection/{inspection_id}_0.jpg"
        image_path2 = f"full_image_inspection/{inspection_id}_2.jpg"
        # Insert the inspection record with image paths
        conn, cursor = get_db()
        cursor.execute(
            "UPDATE Inspection SET image_path1 = ?, image_path2 = ? WHERE inspection_id = ?",
            (image_path1, image_path2, inspection_id)
        )
        conn.commit()


        # check if camera blocked
        camera1_blocked1 = check_camera_blockage(image1,ref_img1)
        camera2_blocked2 = check_camera_blockage(image2,ref_img2)
        is_black_or_white_screen1 = is_black_or_white_screen(image1)
        is_black_or_white_screen2 = is_black_or_white_screen(image2)

        # if(camera1_blocked1 or camera2_blocked2 or is_black_or_white_screen1 or is_black_or_white_screen2):
            # log this
            # logger_main.error("Camera blocked or scene changed")
            # await websocket.send_json({"error": "Camera blocked or scene change detected",
            #                            "current_image1": base64.b64encode(img1).decode("utf-8"),
            #                            "current_image2": base64.b64encode(img2).decode("utf-8")
            #                            })
            # conn.close()
            # raise RuntimeError("Camera blocked or scene changed")

        
        # Process images asynchronously
        bounding_boxes1, bounding_boxes2 = await process_images(image1, image2)
        
        roi_data = {}
        if os.path.exists("roi_data.json"):
            with open("roi_data.json", "r") as f:
                roi_data = json.load(f)
        
        # Convert bounding boxes to required format
        bounding_boxes1 = [{
            "x": x + roi_data["0"]["x"],
            "y": y + roi_data["0"]["y"],
            "width": w,
            "height": h
        } for (x, y, w, h) in bounding_boxes1]
        
        bounding_boxes2 = [{
            "x": x + roi_data["2"]["x"],
            "y": y + roi_data["2"]["y"],
            "width": w,
            "height": h
        } for (x, y, w, h) in bounding_boxes2]
        
        # Store bounding boxes in database
        for bbox in bounding_boxes1:
            cursor.execute(
                "INSERT INTO BoundingBox_AI (inspection_id, camera_index, x_min, y_min, x_max, y_max, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (inspection_id, 0, bbox["x"], bbox["y"], bbox["width"]+bbox['x'], bbox["height"]+bbox['y'], 0.0)
            )
        
        for bbox in bounding_boxes2:
            cursor.execute(
                "INSERT INTO BoundingBox_AI (inspection_id, camera_index, x_min, y_min, x_max, y_max, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (inspection_id, 1, bbox["x"], bbox["y"], bbox["width"]+bbox['x'], bbox["height"]+bbox['y'], 0.0)
            )
        
        conn.commit()
        conn.close()


        
        # Send processed bounding boxes to UI
        await websocket.send_json({
            "current_image1": base64.b64encode(img1).decode("utf-8"),
            "current_image2": base64.b64encode(img2).decode("utf-8"),
            "bounding_boxes1": bounding_boxes1,
            "bounding_boxes2": bounding_boxes2,
            "inspection_id": inspection_id
        })

    except Exception as e:
        logger_main.error(f"Error during inspection process: {str(e)}")
        await websocket.send_json({"error": f"Inspection error: {str(e)}"})
    finally:
        logger_main.info("Closing WebSocket connection")
        await websocket.close()


#-------------------------------------------------------------------------------------------------------------------------------
def capture_image(camera_id=0):
    """Captures an image from the specified camera and handles disconnection properly."""
    try:
        logger_main.info(f"Attempting to capture image from Camera {camera_id}")
        
        camera_url = "rtsp://192.168.10.92/live1.sdp" if camera_id == 0 else "rtsp://192.168.11.93/live1.sdp"

        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            logger_main.error(f"Camera {camera_id} is disconnected or unavailable.")
            return None  # Camera not available

        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger_main.error(f"Failed to capture image from Camera {camera_id}")
            return None

        logger_main.info(f"Successfully captured image from Camera {camera_id}")
        return frame

    except Exception as e:
        logger_main.error(f"Error capturing image from Camera {camera_id}: {e}")
        return None



ROI_FILE = "roi_data.json"
IMAGE_FOLDER = "full_image_setROI"
BACKUP_FOLDER = "backup_images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class SubmitROIRequest(BaseModel):
    camera_id: int
    bounding_box: BoundingBox

def move_old_images():
    """Move previous images to a backup folder before saving new ones."""
    for file in os.listdir(IMAGE_FOLDER):
        if file.endswith(".jpg"):
            shutil.move(os.path.join(IMAGE_FOLDER, file), os.path.join(BACKUP_FOLDER, file))
            logger_main.info(f"Moved old image {file} to backup.")


def save_roi_data(camera_id, bounding_box):
    """Save ROI data for a given camera."""
    try:
        logger_main.info(f"Saving ROI data for Camera {camera_id}")

        # Load existing ROI data
        if os.path.exists(ROI_FILE):
            try:
                with open(ROI_FILE, "r") as f:
                    roi_data = json.load(f)
            except json.JSONDecodeError:
                logger_main.warning("ROI file exists but is corrupted. Resetting data.")
                roi_data = {}
        else:
            roi_data = {}

        # Update ROI data
        roi_data[str(camera_id)] = {
            "x": bounding_box.x,
            "y": bounding_box.y,
            "width": bounding_box.width,
            "height": bounding_box.height
        }

        # Save to file
        with open(ROI_FILE, "w") as f:
            json.dump(roi_data, f, indent=4)

        logger_main.info(f"ROI data saved successfully for Camera {camera_id}")

    except Exception as e:
        logger_main.error(f"Error saving ROI data for Camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save ROI data")

@app.get("/setRoi")
async def set_roi():
    """Capture images from both cameras, save them, and return them to the UI."""
    logger_main.info("set_roi endpoint called.")

    try:
        move_old_images()  # Move old images before saving new ones

        # Capture images from cameras 0 and 2
        frame1 = capture_image(0)
        frame2 = capture_image(2)

        if frame1 is None or frame2 is None:
            raise ValueError("Failed to capture images from one or both cameras.")

        # Encode images in base64 for UI display
        _, im_arr1 = cv2.imencode('.jpg', frame1)
        im1_b64 = base64.b64encode(im_arr1.tobytes()).decode("utf-8")

        _, im_arr2 = cv2.imencode('.jpg', frame2)
        im2_b64 = base64.b64encode(im_arr2.tobytes()).decode("utf-8")

        # Save images with timestamp
        timestamp = datetime.now(swiss_tz).strftime('%Y-%m-%d_%H-%M-%S')
        img1_path = os.path.join(IMAGE_FOLDER, f"0_{timestamp}.jpg")
        img2_path = os.path.join(IMAGE_FOLDER, f"2_{timestamp}.jpg")

        cv2.imwrite(img1_path, frame1)
        cv2.imwrite(img2_path, frame2)

        logger_main.info(f"Images saved successfully: {img1_path}, {img2_path}")

        return {"frame1": im1_b64, "frame2": im2_b64}

    except ValueError as ve:
        logger_main.error(f"ValueError: {ve}")
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger_main.error(f"Unexpected error in set_roi: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/setRoi/submit")
async def submit_roi(data: SubmitROIRequest):
    """Load the latest saved image, crop it, and save the ROI."""
    try:
        logger_main.info(f"Received submit_roi request for Camera {data.camera_id}")

        # Find the latest saved image for the given camera
        images = sorted(
            [f for f in os.listdir(IMAGE_FOLDER) if f.startswith(f"{data.camera_id}_")],
            key=lambda x: datetime.strptime("_".join(x.split("_")[1:3]).split(".")[0], '%Y-%m-%d_%H-%M-%S'),
            reverse=True
        )       
        if not images:
            raise HTTPException(status_code=404, detail="No saved images found for this camera")

        latest_image_path = os.path.join(IMAGE_FOLDER, images[0])
        logger_main.info(f"Using latest image: {latest_image_path}")

        # Load the latest image
        frame = cv2.imread(latest_image_path)
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to load saved image")

        # Crop ROI
        x, y, w, h = data.bounding_box.x, data.bounding_box.y, data.bounding_box.width, data.bounding_box.height
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            raise HTTPException(status_code=400, detail="Invalid bounding box dimensions")

        # Save cropped ROI
        roi_filename = f"roi_camera_{data.camera_id}.jpg"
        roi_path = os.path.join(IMAGE_FOLDER, roi_filename)
        cv2.imwrite(roi_filename, roi)
        logger_main.info(f"ROI image saved: {roi_filename}")

        save_roi_data(data.camera_id, data.bounding_box)
        logger_main.info(f"ROI data saved for Camera {data.camera_id}")

        return {"status": "Success", "roi_image_path": roi_path}
    

    except HTTPException as http_exc:
        logger_main.error(f"HTTP Exception in submit_roi: {http_exc.detail}")
        raise http_exc

    except Exception as e:
        logger_main.exception(f"Unexpected error in submit_roi: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


#--------------------------------------------------------------------------------------------------------------------

THRESHOLD_FILE = "threshold.json"

@app.post("/setThreshold")
async def set_threshold(data: dict):
    """Handles setting a threshold value and saving it to a JSON file."""
    try:
        logger_main.info("Received request to set threshold")

        threshold = data.get("threshold")
        if threshold is None or not isinstance(threshold, float):
            logger_main.warning(f"Invalid threshold value received: {threshold}")
            raise HTTPException(status_code=400, detail="Invalid threshold value")

        # Save threshold value to JSON file
        threshold_data = {"threshold": threshold}
        with open(THRESHOLD_FILE, "w") as f:
            json.dump(threshold_data, f)

        logger_main.info(f"Threshold successfully set to {threshold} and saved to {THRESHOLD_FILE}")

        return {"status": "Success"}

    except HTTPException as http_exc:
        logger_main.error(f"HTTP Exception in set_threshold: {http_exc.detail}")
        raise http_exc  # Reraise the HTTPException so FastAPI handles it properly

    except Exception as e:
        logger_main.exception(f"Unexpected error in set_threshold: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")



#----------------------------------------------------------------------------------------------------------------------

 ## Session APIs


# Pydantic model for session request
class SessionRequest(BaseModel):
    username: str
    role: str
    lot_number: Optional[int] = -1 # Default value for lot_number if not provided by user


@app.post("/newSession")
async def start_session(session_data: SessionRequest):
    """Handles starting a new session for a user."""
    try:
        logger_main.info(f"Received request to start a session for user: {session_data.username}")

        conn, cursor = get_db()

        # Check if user exists, otherwise insert
        cursor.execute("SELECT id FROM Users WHERE username=?", (session_data.username,))
        user = cursor.fetchone()
        if user is None:
            logger_main.info(f"User {session_data.username} not found, creating new user entry.")
            cursor.execute("INSERT INTO Users (username, role) VALUES (?, ?)", (session_data.username, session_data.role))
            conn.commit()
            user_id = cursor.lastrowid
            logger_main.info(f"New user created with ID {user_id}")
        else:
            user_id = user[0]
            logger_main.info(f"User {session_data.username} found with ID {user_id}")

        # Generate session name based on timestamp
        session_name = f"Session_{datetime.now(swiss_tz).strftime('%Y%m%d_%H%M%S')}"
        logger_main.info(f"Generated session name: {session_name}")

        # Start new session
        cursor.execute("INSERT INTO Session (session_name, user_id, lot_number) VALUES (?, ?, ?)", (session_name, user_id, session_data.lot_number))
        conn.commit()

        session_id = cursor.lastrowid  # Get the session ID of the newly created session
        logger_main.info(f"New session started with ID {session_id} for user {session_data.username}")

        return {
            "status": "Success",
            "message": f"Session started for {session_data.username}",
            "session_id": session_id,
            "session_name": session_name
        }

    except sqlite3.Error as db_error:
        logger_main.error(f"Database error while starting session: {db_error}")
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger_main.exception(f"Unexpected error in start_session: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

    finally:
        conn.close()
        logger_main.info("Database connection closed.")



class EndSessionRequest(BaseModel):
    cam1_final_status: int
    cam2_final_status: int
    comment: str

@app.post("/endSession/{session_id}")
async def end_session(session_id: int, request: EndSessionRequest):
    """Handles ending a session."""
    try:
        logger_main.info(f"Received request to end session {session_id} with cam1 status {request.cam1_final_status}, cam2 status {request.cam2_final_status}, and comment: {request.comment}")

        conn, cursor = get_db()

        # Check if session exists and is active
        cursor.execute("SELECT session_id FROM Session WHERE session_id=? AND end_time IS NULL", (session_id,))
        session = cursor.fetchone()
        if session is None:
            logger_main.warning(f"Session {session_id} not found or already ended.")
            raise HTTPException(status_code=404, detail="Session not found or already ended")

        # Update session with end time and final statuses
        end_time = datetime.now(swiss_tz)
        cursor.execute("""
            UPDATE Session 
            SET end_time=?, cam1_final_status=?, cam2_final_status=?, comment=?
            WHERE session_id=?
        """, (end_time, request.cam1_final_status, request.cam2_final_status, request.comment, session_id))
        conn.commit()
        
        logger_main.info(f"Session {session_id} ended successfully at {end_time}")

        return {"status": "Success", "message": f"Session {session_id} ended successfully"}

    except sqlite3.Error as db_error:
        logger_main.error(f"Database error while ending session {session_id}: {db_error}")
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger_main.exception(f"Unexpected error in end_session for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

    finally:
        conn.close()
        logger_main.info("Database connection closed.")


@app.get("/session/{session_id}")
async def get_session(session_id: int):
    """Retrieves session details."""
    try:
        logger_main.info(f"Received request to fetch session details for session ID {session_id}")

        conn, cursor = get_db()

        cursor.execute("SELECT * FROM Session WHERE session_id=?", (session_id,))
        session = cursor.fetchone()

        if session is None:
            logger_main.warning(f"Session {session_id} not found.")
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = {
            "session_id": session[0],
            "session_name": session[1],
            "user_id": session[2],
            "start_time": session[3],
            "end_time": session[4],
            "final_status": session[5]
        }

        logger_main.info(f"Successfully retrieved session details for session ID {session_id}")
        return session_data

    except sqlite3.Error as db_error:
        logger_main.error(f"Database error while retrieving session {session_id}: {db_error}")
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger_main.exception(f"Unexpected error in get_session for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

    finally:
        conn.close()
        logger_main.info("Database connection closed.")


##-----------------------------------------------------------------------------------------------------

class BoundingBoxData(BaseModel):
    x: int
    y: int
    width: int
    height: int
    type: str  # 'FP' for False Positive, 'FN' for False Negative
    comment: Optional[str] = None

class FeedbackItem(BaseModel):
    camera_index: int
    reviewer_comment: str
    bounding_boxes: List[BoundingBoxData]  # Single list for both FP and FN
    false_positives: int
    false_negatives: int

class SubmitFeedbackRequest(BaseModel):
    inspection_id: int
    reviewer_id: int
    feedback: List[FeedbackItem]


@app.post("/submit")
async def submit_feedback(request: SubmitFeedbackRequest):
    """Handles submission of reviewer feedback, including false positives and false negatives in a single table."""
    try:
        logger_main.info(f"Received feedback submission for inspection ID {request.inspection_id} by reviewer {request.reviewer_id}")

        conn, cursor = get_db()

        # Check if the inspection exists
        cursor.execute("SELECT * FROM Inspection WHERE inspection_id = ?", (request.inspection_id,))
        if cursor.fetchone() is None:
            logger_main.warning(f"Inspection ID {request.inspection_id} not found.")
            raise HTTPException(status_code=404, detail="Inspection not found")

        # Check if the reviewer exists
        cursor.execute("SELECT * FROM Users WHERE id = ?", (request.reviewer_id,))
        if cursor.fetchone() is None:
            logger_main.warning(f"Reviewer ID {request.reviewer_id} not found.")
            raise HTTPException(status_code=404, detail="Reviewer ID not found")

        # Insert feedback, false positives, and false negatives
        for feedback in request.feedback:
            logger_main.info(f"Inserting feedback for camera {feedback.camera_index} with comment: {feedback.reviewer_comment}")
            
            cursor.execute(
                "INSERT INTO Reviewer_Feedback (inspection_id, camera_index, reviewer_id, reviewer_comment, Anomalies_found, Anomalies_missed) VALUES (?, ?, ?, ?, ?, ?)",
                (request.inspection_id, feedback.camera_index, request.reviewer_id, feedback.reviewer_comment, feedback.false_positives, feedback.false_negatives)
            )
            feedback_id = cursor.lastrowid
            logger_main.info(f"Inserted feedback with ID {feedback_id} for camera {feedback.camera_index}")

            # Insert False Annotations (FP and FN)
            for bbox in feedback.bounding_boxes:
                logger_main.info(f"Inserting {bbox.type} for camera {feedback.camera_index}: {bbox}")
                cursor.execute(
                    "INSERT INTO False_Annotations (inspection_id, camera_index, x_min, y_min, width, height, type, comment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (request.inspection_id, feedback.camera_index, bbox.x, bbox.y, bbox.width, bbox.height, bbox.type, bbox.comment)
                )

        conn.commit()
        logger_main.info(f"Feedback successfully submitted for inspection ID {request.inspection_id}")
        return {"status": "Success"}

    except sqlite3.Error as db_error:
        logger_main.error(f"Database error while submitting feedback: {db_error}")
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger_main.exception(f"Unexpected error in submit_feedback: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

    finally:
        conn.close()
        logger_main.info("Database connection closed.")




##------------------------------------------------------------------------------------------------------------------------------------
## History API

from fastapi import FastAPI, Query

@app.get("/history")
def get_history(
    page: int = Query(1, alias="page", ge=0),
    page_size: int = Query(10, alias="page_size", ge=1, le=100)
):
    """Fetches paginated session history with final status."""
    logger_main.info(f"Fetching session history - Page: {page}, Page Size: {page_size}")

    try:
        conn, cursor = get_db()
        offset = (page - 1) * page_size

        # Fetch paginated sessions with final_status
        cursor.execute("""
            SELECT s.session_id, s.session_name, u.username, s.start_time, s.end_time, s.cam1_final_status, s.cam2_final_status, s.comment, s.lot_number
            FROM Session s
            JOIN Users u ON s.user_id = u.id
            ORDER BY s.start_time DESC
            LIMIT ? OFFSET ?
        """, (page_size, offset))

        sessions = cursor.fetchall()
        logger_main.info(f"Retrieved {len(sessions)} sessions from database.")

        # Get total session count for pagination
        cursor.execute("SELECT COUNT(*) FROM Session")
        total_sessions = cursor.fetchone()[0]
        logger_main.info(f"Total sessions count: {total_sessions}")

        history_data = []
        for session in sessions:
            session_id, session_name, username, start_time, end_time, cam1_final_status, cam2_final_status, comment, lot_number = session

            status_text = "Completed" if end_time else "In Progress"
            history_data.append({
                "session_id": session_id,
                "session_name": session_name,
                "inspector": username,
                "date_time": start_time,
                "cam1_final_status": cam1_final_status,
                "cam2_final_status": cam2_final_status,
                "comment": comment,  # Now including final_status field
                "status_text": status_text,  # Human-readable status
                "lot_number": lot_number
            })

        total_pages = (total_sessions + page_size - 1) // page_size

        logger_main.info(f"Returning history data - Page: {page}, Total Pages: {total_pages}")
        return {
            "page": page,
            "page_size": page_size,
            "total_sessions": total_sessions,
            "total_pages": total_pages,
            "history": history_data
        }

    except sqlite3.Error as db_error:
        logger_main.error(f"Database error while fetching history: {db_error}")
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger_main.exception(f"Unexpected error in get_history: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    
    #Add exception for 422 Unprocessable Entity
    except ValueError as ve:
       logger_main.error(f"ValueError: {ve} page 0 access error")
       raise HTTPException(status_code=422, detail=str(ve))

    finally:
        conn.close()
        logger_main.info("Database connection closed.")


#--------------------------------------------------------------------------------------------------------------------------

DB_PATH = "inspection_system_new5.db"

class SessionRequest(BaseModel):
    session_id: int


def get_inspections(session_id: int):
    """Fetch inspection details including images, AI bounding boxes, false positives/negatives, and reviewer feedback."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get user_id from Session table
    cursor.execute("SELECT user_id FROM Session WHERE session_id = ?", (session_id,))
    session_data = cursor.fetchone()
    user_id = session_data[0] if session_data else None
    
    cursor.execute("""
        SELECT inspection_id, inspection_timestamp, image_path1, image_path2 
        FROM Inspection 
        WHERE session_id = ?
    """, (session_id,))
    
    inspections = []
    for row in cursor.fetchall():
        inspection_id, timestamp, img_path1, img_path2 = row
        
        def encode_image(img_path):
            try:
                with open(img_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            except Exception:
                return None  # Handle missing images gracefully
        
        # Fetch AI-detected bounding boxes
        cursor.execute("""
            SELECT bbox_id, camera_index, x_min, y_min, x_max, y_max, confidence 
            FROM BoundingBox_AI 
            WHERE inspection_id = ?
        """, (inspection_id,))
        bounding_boxes_ai = [
            {
                "bbox_id": bbox[0],
                "camera_index": bbox[1],
                "x": bbox[2],
                "y": bbox[3],
                "width": bbox[4] - bbox[2],
                "height": bbox[5] - bbox[3],
                "confidence": bbox[6],
                "label": "AI detected"
            }
            for bbox in cursor.fetchall()
        ]
        
        # Fetch False Positives & False Negatives bounding boxes
        cursor.execute("""
            SELECT fa_id, camera_index, x_min, y_min, width, height, type 
            FROM False_Annotations 
            WHERE inspection_id = ?
        """, (inspection_id,))
        
        false_positives_bboxes = {0: [], 1: []}
        false_negatives_bboxes = {0: [], 1: []}
        false_counts = {0: {"false_positives": 0, "false_negatives": 0}, 1: {"false_positives": 0, "false_negatives": 0}}
        
        for bbox in cursor.fetchall():
            bbox_data = {
                "bbox_id": bbox[0],
                "camera_index": bbox[1],
                "x": bbox[2],
                "y": bbox[3],
                "width": bbox[4],
                "height": bbox[5],
                "label": "False Positive" if bbox[6] == "FP" else "False Negative"
            }
            if bbox[6] == "FP":
                false_positives_bboxes[bbox[1]].append(bbox_data)
                false_counts[bbox[1]]["false_positives"] += 1
            elif bbox[6] == "FN":
                false_negatives_bboxes[bbox[1]].append(bbox_data)
                false_counts[bbox[1]]["false_negatives"] += 1
        
        # Reviewer feedback
        cursor.execute("""
            SELECT feedback_id, reviewer_comment, Anomalies_found, Anomalies_missed
            FROM Reviewer_Feedback 
            WHERE inspection_id = ?
        """, (inspection_id,))
        reviewer_feedback = cursor.fetchone()
        feedback_data = {
            "feedback_id": reviewer_feedback[0],
            "reviewer_comment": reviewer_feedback[1],
            "false_positives": reviewer_feedback[2],
            "false_negatives": reviewer_feedback[3]
        } if reviewer_feedback else None
        
        inspections.append({
            "inspection_id": inspection_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "image1": encode_image(img_path1),
            "image2": encode_image(img_path2),
            "bounding_boxes_ai": bounding_boxes_ai,
            "false_positives_bboxes": false_positives_bboxes,
            "false_negatives_bboxes": false_negatives_bboxes,
            "false_counts_per_camera": false_counts,
            "total_detections": len(bounding_boxes_ai),
            "reviewer_feedback": feedback_data
        })
    
    conn.close()
    return inspections




@app.post("/inspections")
def get_inspections_by_session(request: SessionRequest):
    logger_main.info(f"Fetching inspections for session ID {request.session_id}")
    inspections = get_inspections(request.session_id)
    if not inspections:
        logger_main.warning(f"No inspections found for session ID {request.session_id}")
        raise HTTPException(status_code=404, detail="No inspections found for this session")
    logger_main.info(f"Returning {len(inspections)} inspections for session ID {request.session_id}")
    return {"session_id": request.session_id, "inspections": inspections}

#-----------------------------------------------------------------------------------------------------------------------------------

class LogoutRequest(BaseModel):
    user_id: int


@app.post("/logout")
async def logout(request: LogoutRequest):
    """Handles user logout by marking user as logged out but keeping the session open."""
    try:
        logger_main.info(f"Logging out user with ID: {request.user_id}")
        # conn, cursor = get_db()

        # # Optionally mark the user as logged out
        # cursor.execute("UPDATE Users SET is_logged_in=0 WHERE id=?", (user_id,))
        # conn.commit()

        logger_main.info(f"User {request.user_id} logged out successfully")
        return {"status": "Success", "message": "Logout successful. You can log in again anytime."}
    
    # except sqlite3.Error as db_error:
    #     logger_main.error(f"Database error during logout: {db_error}")
    #     raise HTTPException(status_code=500, detail="Database error occurred")
    
    finally:
        # conn.close()
        # logger_main.info("Database connection closed.")
        pass



# # Paths to the required directories and files
# IMAGE_DIRECTORY = "full_image_setROI"  # Change this to the actual directory path
# ROI_JSON_PATH = "roi_data.json"
# THRESHOLD_JSON_PATH = "threshold.json"

# def get_images_from_directory(directory):
#     """Finds and sorts two images from the given directory."""
#     try:
#         if not os.path.exists(directory):
#             raise FileNotFoundError(f"Directory not found: {directory}")
        
#         # List all files, filter for images, and sort them
#         image_files = sorted(
#             [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#         )
        
#         if len(image_files) < 2:
#             raise ValueError("Not enough images found in the directory (need at least 2).")

#         return os.path.join(directory, image_files[0]), os.path.join(directory, image_files[1])

#     except Exception as e:
#         return str(e), None  # Return an error string if something goes wrong

# def encode_image(image_path):
#     """Converts an image to base64 encoding."""
#     try:
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"File not found: {image_path}")
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode("utf-8")
#     except Exception as e:
#         return str(e)

# def read_json(file_path):
#     """Reads JSON file and handles errors."""
#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#         with open(file_path, "r") as json_file:
#             return json.load(json_file)
#     except json.JSONDecodeError:
#         return {"error": f"Invalid JSON format in {file_path}"}
#     except Exception as e:
#         return {"error": str(e)}

# @app.get("/get_roi_data")
# def get_data():
#     errors = []

#     move_old_images()  # Move old images before saving new ones

#     # Capture images from cameras 0 and 2
#     frame1 = capture_image(0)
#     frame2 = capture_image(2)

#     if frame1 is None or frame2 is None:
#         raise ValueError("Failed to capture images from one or both cameras.")

#     # Encode images in base64 for UI display
#     _, im_arr1 = cv2.imencode('.jpg', frame1)
#     im1_b64 = base64.b64encode(im_arr1.tobytes()).decode("utf-8")

#     _, im_arr2 = cv2.imencode('.jpg', frame2)
#     im2_b64 = base64.b64encode(im_arr2.tobytes()).decode("utf-8")

#     # Save images with timestamp
#     timestamp = datetime.now(swiss_tz).strftime('%Y-%m-%d_%H-%M-%S')
#     img1_path = os.path.join(IMAGE_FOLDER, f"0_{timestamp}.jpg")
#     img2_path = os.path.join(IMAGE_FOLDER, f"2_{timestamp}.jpg")

#     cv2.imwrite(img1_path, frame1)
#     cv2.imwrite(img2_path, frame2)

#     logger_main.info(f"Images saved successfully: {img1_path}, {img2_path}")


#     # Read JSON data
#     roi_data = read_json(ROI_JSON_PATH)
#     if "error" in roi_data:
#         errors.append(roi_data["error"])

#     threshold_data = read_json(THRESHOLD_JSON_PATH)
#     if "error" in threshold_data:
#         errors.append(threshold_data["error"])

#     # If there are errors, return them
#     if errors:
#         return JSONResponse(content={"errors": errors}, status_code=500)

#     # Construct response
#     response_data = {
#         "images": {
#             "image1": im1_b64,
#             "image2": im2_b64
#         },
#         "roi": roi_data,
#         "threshold": threshold_data
#     }

#     return response_data

#-------------------------------------------------------------------------------------------------------------------------------

IMAGE_DIRECTORY = "full_image_setROI"  # Change this to the actual directory path
ROI_JSON_PATH = "roi_data.json"
THRESHOLD_JSON_PATH = "threshold.json"

def get_images_from_directory(directory):
    """Finds and sorts two images from the given directory."""
    try:
        if not os.path.exists(directory):
            logger_main.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")

        image_files = sorted(
            [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        if len(image_files) < 2:
            logger_main.warning(f"Not enough images found in {directory}. Found: {len(image_files)}")
            raise ValueError("Not enough images found in the directory (need at least 2).")

        logger_main.info(f"Found images: {image_files[:2]}")
        return os.path.join(directory, image_files[0]), os.path.join(directory, image_files[1])

    except Exception as e:
        logger_main.exception(f"Error in get_images_from_directory: {e}")
        return str(e), None


def encode_image(image_path):
    """Converts an image to base64 encoding."""
    try:
        if not os.path.exists(image_path):
            logger_main.error(f"File not found: {image_path}")
            raise FileNotFoundError(f"File not found: {image_path}")

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            logger_main.info(f"Image encoded successfully: {image_path}")
            return encoded

    except Exception as e:
        logger_main.exception(f"Error in encode_image: {e}")
        return str(e)


def read_json(file_path):
    """Reads JSON file and handles errors."""
    try:
        if not os.path.exists(file_path):
            logger_main.warning(f"JSON file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            logger_main.info(f"Successfully read JSON: {file_path}")
            return data

    except json.JSONDecodeError:
        logger_main.error(f"Invalid JSON format in {file_path}")
        return {"error": f"Invalid JSON format in {file_path}"}
    
    except Exception as e:
        logger_main.exception(f"Error reading JSON: {file_path}, Error: {e}")
        return {"error": str(e)}


@app.get("/get_roi_data")
def get_data():
    """Fetches ROI data, threshold data, and captures images."""
    errors = []

    logger_main.info("get_roi_data API called")

    move_old_images()  # Move old images before saving new ones

    # Capture images from cameras
    frame1 = capture_image(0)
    frame2 = capture_image(2)

    if frame1 is None or frame2 is None:
        logger_main.error("Failed to capture images from one or both cameras")
        raise HTTPException(status_code=500, detail="Failed to capture images from one or both cameras.")

    logger_main.info("Successfully captured images from both cameras")

    # Encode images
    try:
        _, im_arr1 = cv2.imencode('.jpg', frame1)
        im1_b64 = base64.b64encode(im_arr1.tobytes()).decode("utf-8")

        _, im_arr2 = cv2.imencode('.jpg', frame2)
        im2_b64 = base64.b64encode(im_arr2.tobytes()).decode("utf-8")

        logger_main.info("Images successfully encoded to base64")
    except Exception as e:
        logger_main.exception(f"Error encoding images: {e}")
        raise HTTPException(status_code=500, detail="Error encoding images")

    # Save images
    timestamp = datetime.now(swiss_tz).strftime('%Y-%m-%d_%H-%M-%S')
    img1_path = os.path.join(IMAGE_FOLDER, f"0_{timestamp}.jpg")
    img2_path = os.path.join(IMAGE_FOLDER, f"2_{timestamp}.jpg")

    try:
        cv2.imwrite(img1_path, frame1)
        cv2.imwrite(img2_path, frame2)
        logger_main.info(f"Images saved successfully: {img1_path}, {img2_path}")
    except Exception as e:
        logger_main.exception(f"Error saving images: {e}")
        raise HTTPException(status_code=500, detail="Error saving images")

    # Read JSON data
    roi_data = read_json(ROI_JSON_PATH)
    if "error" in roi_data:
        errors.append(roi_data["error"])

    threshold_data = read_json(THRESHOLD_JSON_PATH)
    if "error" in threshold_data:
        errors.append(threshold_data["error"])

    if errors:
        logger_main.warning(f"Errors occurred: {errors}")
        return JSONResponse(content={"errors": errors}, status_code=500)

    # Construct response
    response_data = {
        "images": {
            "image1": im1_b64,
            "image2": im2_b64
        },
        "roi": roi_data,
        "threshold": threshold_data
    }

    logger_main.info("Successfully fetched ROI and threshold data")
    return response_data
