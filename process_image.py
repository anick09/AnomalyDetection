
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

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Add at top of file after imports
if not os.path.exists("logs"):
    os.makedirs("logs")

swiss_tz = pytz.timezone('Europe/Zurich')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now(swiss_tz).strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def check_camera_blockage(reference_image, current_image, threshold=0.71):
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(ref_gray, curr_gray, full=True)
    print(f"SSIM Score: {score}")
    return score < threshold  # True means possible blockage

import cv2
import numpy as np

def is_black_or_white_screen(image, std_threshold=10, mean_threshold=50):
    if image is None:
        raise ValueError("Image is empty or not loaded properly")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    mean_pixel = np.mean(gray)  # Average intensity
    std_pixel = np.std(gray)  # Standard deviation of intensity
    
    print(f"Mean: {mean_pixel:.2f}, Std Dev: {std_pixel:.2f}")

    # Black screen: Low mean and low variation
    if mean_pixel < mean_threshold and std_pixel < std_threshold:
        return True  # Likely a black screen

    # White screen: High mean and low variation
    if mean_pixel > (255 - mean_threshold) and std_pixel < std_threshold:
        return True  # Likely a white screen

    return False



def extract_embeddings(model, image, boxes, device):
    # img=image.detach().cpu().numpy()
    # img=img[0].transpose(1,2,0)
    # print(type(img))
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(boxes)

    embeddings, positions = [], []
    for (x, y, w, h) in boxes:
        window = image[:, :, y:y+h, x:x+w]
        img=window.detach().cpu().numpy()
        img=img[0].transpose(1,2,0)
        # cv2.imshow("window", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        with torch.no_grad():
            embedding = model(window).squeeze().cpu().numpy()
        embeddings.append(embedding)
        positions.append((x, y, w, h))

    print(f"embeddings extracted for {len(embeddings)} boxes")
    return np.array(embeddings), positions

def compute_similarity_scores(ref_embeddings, tgt_embeddings, metric, cov_matrix=None):
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
        raise ValueError("Unsupported similarity metric")

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

    anomaly_windows = [positions[i] for i, score in enumerate(scores) if score < 0.7]
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

def visualize_final_anomalies(target_img_path, final_boxes,camid):
    target_full_img= cv2.imread(target_img_path)
    if(camid==0):
        target_full_img= cv2.imread("/home/sr/jnjLineclearing/fastapi-sqlite-jnjbackend/full_image_inspection/26_0.jpg")
    else:
        target_full_img= cv2.imread("/home/sr/jnjLineclearing/fastapi-sqlite-jnjbackend/full_image_inspection/26q_2.jpg")

    roi_data = {}
    import os,json
    if os.path.exists("roi_data.json"):
        with open("roi_data.json", "r") as f:
            roi_data = json.load(f)
    
    #  roi_data is in format
    # {
    # "0": {
    #     "x": 51,
    #     "y": 141,
    #     "width": 206,
    #     "height": 327
    # },
    # "2": {
    #     "x": 100,
    #     "y": 50,
    #     "width": 400,
    #     "height": 350
    # }
    # }
    # Now map the bounding_boxes
    # bounding_boxes1 = [(x+roi_data["0"]["x"], y+roi_data["0"]["y"], w, h) for (x, y, w, h) in bounding_boxes1]
    # bounding_boxes2 = [(x+roi_data["2"]["x"], y+roi_data["2"]["y"], w, h) for (x, y, w, h) in bounding_boxes2]
    # target_img = cv2.imread()

    #based on this make changes on the final_boxes
    final_boxes = [(x+roi_data[str(camid)]["x"], y+roi_data[str(camid)]["y"], w, h) for (x, y, w, h) in final_boxes]



    for (x, y, w, h) in final_boxes:
        cv2.rectangle(target_full_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if(target_full_img):
        cv2.imshow("Final Anomalies", target_full_img)
        cv2.imwrite(f"final_anomalies_filtered_{camid}.jpg", target_full_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"final Boxes Count : {len(final_boxes)}")

def find_anomaly(reference_image_path, target_image_path, camid, roi, metric='cosine', threshold_method='fixed'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    reference_image_full = cv2.imread(reference_image_path)
    # cv2.imshow("reference_image", reference_image_full)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    target_image_full = cv2.imread(target_image_path)
    # cv2.imshow("target_image", target_image_full)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print("Select ROI and press ENTER")
    # roi = cv2.selectROI("Select ROI", reference_image_full, showCrosshair=True)
    # cv2.destroyWindow("Select ROI")

    #check if camera is blocked
    # if check_camera_blockage(reference_image_full, target_image_full):
    #     print("Camera is blocked")
    #     raise RuntimeError("Camera is blocked")



    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    
    # reference_roi = reference_image_full[y:y+h, x:x+w]
    # target_roi = target_image_full[y:y+h, x:x+w]
    
    ssim_start_time=time.time()

    ssim_boxes = get_ssim_bounding_boxes(reference_image_full, target_image_full)

    # for (bx, by, bw, bh) in ssim_boxes:
    #     print(type(bx), type(x))

    # ssim_boxes = [(x+bx, y+by, bw, bh) for (bx, by, bw, bh) in ssim_boxes]
    ssim_end_time=time.time()

    model_load_start_time=time.time()
    model = resnet50(pretrained=True).to(device)
    # model = resnet18(pretrained=True).to(device)
    # model = resnet101(pretrained=True).to(device)
    # model = resnet34(pretrained=True).to(device)
    # model = resnet152(pretrained=True).to(device)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model_load_end_time=time.time()

    pre_procc_start=time.time()    
    reference_image_tensor = preprocess_image(Image.fromarray(reference_image_full), device)
    target_image_tensor = preprocess_image(Image.fromarray(target_image_full), device)
    pre_procc_end=time.time()
    
    resnet_embed_start_time=time.time()
    resnet_anomalies = get_resnet_anomalies(reference_image_tensor, target_image_tensor, model, ssim_boxes, metric, threshold_method, device)
    resnet_embed_end_time=time.time()
    # visualize_final_anomalies(target_image_path, resnet_anomalies,camid)
    

    print(f"resnet50 time: {resnet_embed_end_time-resnet_embed_start_time:.2f} seconds")
    print(f"total_time : {(ssim_end_time-ssim_start_time)+(pre_procc_end-pre_procc_start)+(resnet_embed_end_time-resnet_embed_start_time):.2f} seconds")

    return resnet_anomalies

# if __name__ == "__main__":
#     reference_image_path = "/home/sr/jnjLineclearing/JNJ_Lineclearing/notebooks/images/frame_1092.jpg"
#     target_image_path = "/home/sr/jnjLineclearing/JNJ_Lineclearing/notebooks/images/frame_1122.jpg"
    
#     main(reference_image_path, target_image_path, metric='cosine', threshold_method='percentile')
