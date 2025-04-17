import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18, resnet152, resnet101, resnet34
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, mahalanobis
from scipy.stats import entropy, wasserstein_distance
import cv2
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans
import time
import warnings
import logging
import os
from datetime import datetime
import pytz 

# Define base Docker volume path
DOCKER_VOLUME_PATH = "/data" 

# Define log directory under Docker volume
LOG_DIR = f"{DOCKER_VOLUME_PATH}/process_image_logs"  # Modified line 24

# Ensure the log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Timezone setup
swiss_tz = pytz.timezone('Europe/Zurich')

# Configure logging
# Unique log file for process_image.py
log_filename = f'{LOG_DIR}/process_image_{datetime.now(pytz.timezone("Europe/Zurich")).strftime("%Y%m%d")}.log'  # Line 35 (unchanged since LOG_DIR is updated)

# Configure a separate logger for process_image.py
logger = logging.getLogger("process_image")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if script is re-imported
if not logger.hasHandlers():
    logging.Formatter.converter = lambda *args: datetime.now(swiss_tz).timetuple()
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

logger.info("Logging initialized in process_image_logs")

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def get_dynamic_threshold(scores, method='percentile', percentile=5):
    if method == 'std_dev':
        return np.mean(scores) - (0.5 * np.std(scores))
    elif method == 'percentile':
        return np.percentile(scores, percentile)
    elif method == 'clustering':
        kmeans = KMeans(n_clusters=2, random_state=42).fit(scores.reshape(-1, 1))
        labels = kmeans.labels_
        return np.max(scores[labels == labels[np.argmax(scores)]])
    elif method == 'fixed':
        threshold_data = {}
        import os, json
        threshold_file = f"{DOCKER_VOLUME_PATH}/threshold.json"  # Modified line 70
        if os.path.exists(threshold_file):  # Modified line 71
            with open(threshold_file, "r") as f:  # Modified line 72
                threshold_data = json.load(f)
        print(f"threshold {threshold_data['threshold']}")
        return threshold_data["threshold"]
    else:
        raise ValueError("Invalid thresholding method")

# Rest of the functions remain unchanged unless they involve file operations
def get_resnet_anomalies(reference_image, target_image, model, ssim_boxes, metric, threshold_method, device):
    embedding_ext_start = time.time()
    ref_embeddings, positions = extract_embeddings(model, reference_image, ssim_boxes, device)
    tgt_embeddings, _ = extract_embeddings(model, target_image, ssim_boxes, device)
    embedding_ext_end = time.time()
    print(f"embedding extraction for both image roi: {embedding_ext_end-embedding_ext_start:.2f} seconds")

    cov_matrix = np.cov(ref_embeddings.T) if metric == 'mahalanobis' else None

    if len(ref_embeddings) == 0 or len(tgt_embeddings) == 0:
        logger.info("Reference or Target embeddings are empty")
        return []
    
    scores = compute_similarity_scores(ref_embeddings, tgt_embeddings, metric, cov_matrix)

    dynamic_thres_start_time = time.time()
    threshold = get_dynamic_threshold(scores, method=threshold_method)
    dynamic_thres_end_time = time.time()
    print(f"dynamic threshold time : {dynamic_thres_end_time-dynamic_thres_start_time:.2f} seconds")
    print(f"Using {metric} similarity, computed threshold: {threshold:.4f}")
    
    for score in scores:
        print(score)

    anomaly_windows = [positions[i] for i, score in enumerate(scores) if score < threshold]
    return anomaly_windows

def get_ssim_bounding_boxes(reference_roi, target_roi):
    ref_gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    
    score, diff = structural_similarity(ref_gray, tgt_gray, full=True)
    print(f"Image Similarity: {score * 100:.2f}%")
    
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ssim_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]
    print(f"SSIM Boxes Count : {len(ssim_boxes)}")
    return ssim_boxes

def preprocess_image(image, device):
    logger.info("Preprocessing image for ResNet model")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def check_camera_blockage(reference_image, current_image, threshold=0.75):
    logger.info("Checking for camera blockage using SSIM")
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(ref_gray, curr_gray, full=True)
    
    logger.info(f"SSIM Score: {score:.4f} (Threshold: {threshold})")
    return score < threshold

def is_black_or_white_screen(image, std_threshold=10, mean_threshold=50):
    if image is None:
        logger.warning("Received an empty image for black/white screen detection")
        raise ValueError("Image is empty or not loaded properly")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_pixel = np.mean(gray)
    std_pixel = np.std(gray)

    logger.info(f"Mean: {mean_pixel:.2f}, Std Dev: {std_pixel:.2f}")

    if mean_pixel < mean_threshold and std_pixel < std_threshold:
        logger.warning("Detected a black screen")
        return True  
    if mean_pixel > (255 - mean_threshold) and std_pixel < std_threshold:
        logger.warning("Detected a white screen")
        return True  

    return False

def extract_embeddings(model, image, boxes, device):
    logger.info(f"Extracting embeddings for {len(boxes)} bounding boxes")
    
    embeddings, positions = [], []
    for (x, y, w, h) in boxes:
        window = image[:, :, y:y+h, x:x+w]
        with torch.no_grad():
            embedding = model(window).squeeze().cpu().numpy()
        embeddings.append(embedding)
        positions.append((x, y, w, h))

    logger.info(f"Extracted embeddings for {len(embeddings)} regions")
    return np.array(embeddings), positions

def compute_similarity_scores(ref_embeddings, tgt_embeddings, metric, cov_matrix=None):
    logger.info(f"Computing similarity scores using {metric} metric")
    
    if metric == 'cosine':
        return cosine_similarity(ref_embeddings, tgt_embeddings).diagonal()
    elif metric == 'euclidean':
        return np.array([euclidean(ref, tgt) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
    elif metric == 'mahalanobis' and cov_matrix is not None:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return np.array([mahalanobis(ref, tgt, inv_cov_matrix) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
    elif metric == 'kl_divergence':
        return np.array([entropy(ref, tgt) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
    elif metric == 'wasserstein':
        return np.array([wasserstein_distance(ref, tgt) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
    else:
        logger.error(f"Unsupported similarity metric: {metric}")
        raise ValueError("Unsupported similarity metric")

def find_anomaly(reference_image_path, target_image_path, camid, roi, metric='cosine', threshold_method='fixed'):
    logger.info(f"Starting anomaly detection for Camera {camid}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference_image_full = cv2.imread(reference_image_path)
    target_image_full = cv2.imread(target_image_path)

    if reference_image_full is None or target_image_full is None:
        logger.error("Error: One or both images could not be loaded")
        raise ValueError("Error: One or both images could not be loaded")

    if check_camera_blockage(reference_image_full, target_image_full):
        logger.error("Camera is blocked or scene changed")
        raise RuntimeError("Camera is blocked or scene changed")

    logger.info("Running SSIM-based bounding box extraction")
    ssim_start_time = time.time()
    ssim_boxes = get_ssim_bounding_boxes(reference_image_full, target_image_full)
    ssim_end_time = time.time()
    logger.info(f"SSIM bounding box extraction took {ssim_end_time - ssim_start_time:.2f} seconds")

    # logger.info(f"Loading ResNet model on {device}")
    # model = resnet18(pretrained=True).to(device)
    # model = nn.Sequential(*list(model.children())[:-1])
    # model.eval()
    
    logger.info(f"Loading ResNet model on {device}")
    model_path = f"{DOCKER_VOLUME_PATH}/resnet18.pth"
    if os.path.exists(model_path):
        logger.info("Loading ResNet model from local path")
        model = resnet18(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.error("Model file not found. Please ensure it is available at the specified path.")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()


    logger.info("Preprocessing images")
    reference_image_tensor = preprocess_image(Image.fromarray(reference_image_full), device)
    target_image_tensor = preprocess_image(Image.fromarray(target_image_full), device)

    logger.info("Extracting feature embeddings")
    resnet_embed_start_time = time.time()
    resnet_anomalies = get_resnet_anomalies(
        reference_image_tensor, target_image_tensor, model, ssim_boxes, metric, threshold_method, device
    )
    resnet_embed_end_time = time.time()
    logger.info(f"Feature extraction and anomaly detection took {resnet_embed_end_time - resnet_embed_start_time:.2f} seconds")

    return resnet_anomalies
