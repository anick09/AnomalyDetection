
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50,resnet18,resnet152,resnet101,resnet34
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
import cv2
import numpy as np
import logging
import os
from datetime import datetime
import pytz 

# Define log directory
LOG_DIR = "process_image_logs"

# Ensure the log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Timezone setup
swiss_tz = pytz.timezone('Europe/Zurich')

# Configure logging
# Unique log file for process_image.py
log_filename = f'{LOG_DIR}/process_image_{datetime.now(pytz.timezone("Europe/Zurich")).strftime("%Y%m%d")}.log'

# Configure a separate logger for process_image.py
logger = logging.getLogger("process_image")  # <== Named Logger
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if script is re-imported
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

logger.info("Logging initialized in process_image_logs")



warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")




# def preprocess_image(image, device):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(image).unsqueeze(0).to(device)

# def check_camera_blockage(reference_image, current_image, threshold=0.75):
#     ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
#     score, _ = structural_similarity(ref_gray, curr_gray, full=True)
#     print(f"SSIM Score: {score}")
#     return score < threshold  # True means possible blockage



# def is_black_or_white_screen(image, std_threshold=10, mean_threshold=50):
#     if image is None:
#         raise ValueError("Image is empty or not loaded properly")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     mean_pixel = np.mean(gray)  # Average intensity
#     std_pixel = np.std(gray)  # Standard deviation of intensity
    
#     print(f"Mean: {mean_pixel:.2f}, Std Dev: {std_pixel:.2f}")

#     # Black screen: Low mean and low variation
#     if mean_pixel < mean_threshold and std_pixel < std_threshold:
#         return True  # Likely a black screen

#     # White screen: High mean and low variation
#     if mean_pixel > (255 - mean_threshold) and std_pixel < std_threshold:
#         return True  # Likely a white screen

#     return False



# def extract_embeddings(model, image, boxes, device):

#     embeddings, positions = [], []
#     for (x, y, w, h) in boxes:
#         window = image[:, :, y:y+h, x:x+w]
#         img=window.detach().cpu().numpy()
#         img=img[0].transpose(1,2,0)
#         with torch.no_grad():
#             embedding = model(window).squeeze().cpu().numpy()
#         embeddings.append(embedding)
#         positions.append((x, y, w, h))

#     print(f"embeddings extracted for {len(embeddings)} boxes")
#     return np.array(embeddings), positions

# def compute_similarity_scores(ref_embeddings, tgt_embeddings, metric, cov_matrix=None):
#     if metric == 'cosine':
#         return cosine_similarity(ref_embeddings, tgt_embeddings).diagonal()
#     elif metric == 'euclidean':
#         return np.array([euclidean(ref, tgt) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
#     elif metric == 'mahalanobis' and cov_matrix is not None:
#         inv_cov_matrix = np.linalg.inv(cov_matrix)
#         return np.array([mahalanobis(ref, tgt, inv_cov_matrix) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
#     elif metric == 'kl_divergence':
#         return np.array([entropy(ref, tgt) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
#     elif metric == 'wasserstein':
#         return np.array([wasserstein_distance(ref, tgt) for ref, tgt in zip(ref_embeddings, tgt_embeddings)])
#     else:
#         raise ValueError("Unsupported similarity metric")

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
        import os,json
        if os.path.exists("threshold.json"):
            with open("threshold.json", "r") as f:
                threshold_data = json.load(f)
        print(f"threshold {threshold_data['threshold']}")
        return threshold_data["threshold"]
    else:
        raise ValueError("Invalid thresholding method")

def get_resnet_anomalies(reference_image, target_image, model, ssim_boxes, metric, threshold_method, device):
    embedding_ext_start=time.time()
    ref_embeddings, positions = extract_embeddings(model, reference_image, ssim_boxes, device)
    tgt_embeddings, _ = extract_embeddings(model, target_image, ssim_boxes, device)
    embedding_ext_end=time.time()
    print(f"embedding extraction for both image roi: {embedding_ext_end-embedding_ext_start:.2f} seconds")


    cov_matrix = np.cov(ref_embeddings.T) if metric == 'mahalanobis' else None

    #if ref_embeddings is empty or tgt_embeddings is empty return empty list
    if len(ref_embeddings)==0 or len(tgt_embeddings)==0:
        logger.info("Reference or Target embeddings are empty")
        return []
    
    scores = compute_similarity_scores(ref_embeddings, tgt_embeddings, metric, cov_matrix)

    dynamic_thres_start_time=time.time()
    threshold = get_dynamic_threshold(scores, method=threshold_method)
    dynamic_thres_end_time=time.time()
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

# def visualize_final_anomalies(target_img_path, final_boxes,camid):
#     target_full_img= cv2.imread(target_img_path)
#     if(camid==0):
#         target_full_img= cv2.imread("/home/sr/jnjLineclearing/fastapi-sqlite-jnjbackend/full_image_inspection/26_0.jpg")
#     else:
#         target_full_img= cv2.imread("/home/sr/jnjLineclearing/fastapi-sqlite-jnjbackend/full_image_inspection/26q_2.jpg")

#     roi_data = {}
#     import os,json
#     if os.path.exists("roi_data.json"):
#         with open("roi_data.json", "r") as f:
#             roi_data = json.load(f)
    

#     #based on this make changes on the final_boxes
#     final_boxes = [(x+roi_data[str(camid)]["x"], y+roi_data[str(camid)]["y"], w, h) for (x, y, w, h) in final_boxes]



#     for (x, y, w, h) in final_boxes:
#         cv2.rectangle(target_full_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     if(target_full_img):
#         cv2.imshow("Final Anomalies", target_full_img)
#         cv2.imwrite(f"final_anomalies_filtered_{camid}.jpg", target_full_img)
#         cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print(f"final Boxes Count : {len(final_boxes)}")

# def find_anomaly(reference_image_path, target_image_path, camid, roi, metric='cosine', threshold_method='fixed'):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     reference_image_full = cv2.imread(reference_image_path)
#     target_image_full = cv2.imread(target_image_path)

#     # check if camera is blocked
#     if check_camera_blockage(reference_image_full, target_image_full):
#         print("Camera is blocked or scene changed")
#         raise RuntimeError("Camera is blocked or scene changed")



#     x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    
    
#     ssim_start_time=time.time()

#     ssim_boxes = get_ssim_bounding_boxes(reference_image_full, target_image_full)
#     ssim_end_time=time.time()

#     model_load_start_time=time.time()
#     # model = resnet50(pretrained=True).to(device)
#     model = resnet18(pretrained=True).to(device)
#     # model = resnet101(pretrained=True).to(device)
#     # model = resnet34(pretrained=True).to(device)
#     # model = resnet152(pretrained=True).to(device)
#     model = nn.Sequential(*list(model.children())[:-1])
#     model.eval()
#     model_load_end_time=time.time()

#     pre_procc_start=time.time()    
#     reference_image_tensor = preprocess_image(Image.fromarray(reference_image_full), device)
#     target_image_tensor = preprocess_image(Image.fromarray(target_image_full), device)
#     pre_procc_end=time.time()
    
#     resnet_embed_start_time=time.time()
#     resnet_anomalies = get_resnet_anomalies(reference_image_tensor, target_image_tensor, model, ssim_boxes, metric, threshold_method, device)
#     resnet_embed_end_time=time.time()
    

#     print(f"resnet50 time: {resnet_embed_end_time-resnet_embed_start_time:.2f} seconds")
#     print(f"total_time : {(ssim_end_time-ssim_start_time)+(pre_procc_end-pre_procc_start)+(resnet_embed_end_time-resnet_embed_start_time):.2f} seconds")

#     return resnet_anomalies


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

    logger.info(f"Loading ResNet model on {device}")
    model = resnet18(pretrained=True).to(device)
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


