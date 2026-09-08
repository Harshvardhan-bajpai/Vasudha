import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict, Counter
import numpy as np
import cv2
import os
import json
import uuid


# =====================================
# DEVICE
# =====================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================
# PATHS
# =====================================

CROP_CLASSIFIER_PATH = "crop_classifier_trained_model/resnet18_best.pth"
OOD_PATH             = "crop_classifier_trained_model/ood_stats.json"
SENSOR_FILE          = "sensor_values.json"

MODEL_PATHS = {
    "Rice"     : "rice_trained_model/resnet18_best.pth",
    "Wheat"    : "wheat_trained_model/resnet18_best.pth",
    "Sugarcane": "sugarcane_trained_model/resnet18_best.pth",
}


# =====================================
# DISEASE LABELS  (must match training order)
# =====================================

DISEASE_CLASSES = {
    "Rice"     : ["Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf scald"],
    "Wheat"    : ["BlackPoint", "FusariumFootRot", "HealthyLeaf", "LeafBlight", "WheatBlast"],
    "Sugarcane": ["Healthy", "Mosaic", "RedRot", "Yellow"],
}

CROP_CLASSES = ["Rice", "Sugarcane", "Wheat"]

# Healthy keyword variants — covers all naming styles across crop models
HEALTHY_KEYWORDS = {
    "healthy", "healthyleaf", "healthyriceleaf",
    "healthy rice leaf", "healthy leaf"
}


# =====================================
# TUNABLE THRESHOLDS
# =====================================

BLUR_THRESHOLD           = 100.0   # Laplacian variance — below this = blurry
CROP_CONF_MIN            = 70.0    # min crop classifier confidence to accept image
PER_IMAGE_MIN_CONFIDENCE = 40.0    # ignore disease predictions below this %
OUTLIER_CONF_THRESHOLD   = 85.0    # reject disease outlier if high-conf + disagrees with majority
WEIGHTED_VOTE_MIN_SHARE  = 45.0    # final result needs this % of total vote weight
MIN_WEIGHT_FOR_DECISION  = 150.0   # minimum total vote weight to make a call
TOP_K                    = 2       # use top-2 predictions per image
TOP_K_DISCOUNT           = 0.6     # 2nd place gets 60% of its confidence as weight


# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# LOAD CROP CLASSIFIER
# =====================================

crop_classifier = models.resnet18(weights=None)
crop_classifier.fc = nn.Linear(crop_classifier.fc.in_features, len(CROP_CLASSES))
crop_classifier.load_state_dict(torch.load(CROP_CLASSIFIER_PATH, map_location=DEVICE))
crop_classifier = crop_classifier.to(DEVICE)
crop_classifier.eval()


# =====================================
# OOD — Mahalanobis feature hook
# =====================================

feat_cache = {}

def _hook_fn(module, input, output):
    feat_cache["feats"] = output.detach().cpu().squeeze(-1).squeeze(-1).numpy()

_hook_handle = crop_classifier.avgpool.register_forward_hook(_hook_fn)


# =====================================
# LOAD OOD STATS
# =====================================

if os.path.exists(OOD_PATH):
    with open(OOD_PATH) as f:
        _ood = json.load(f)
    OOD_THRESHOLD = _ood["threshold"]
    global_mean   = np.array(_ood["global_mean"])
    cov_inv       = np.array(_ood["cov_inv"])
else:
    OOD_THRESHOLD = 999.0          # effectively disabled if file missing
    global_mean   = None
    cov_inv       = None


# =====================================
# LOAD DISEASE MODELS  (lazy cache)
# =====================================

_disease_models = {}

def _get_disease_model(crop_name):
    if crop_name not in _disease_models:
        path        = MODEL_PATHS[crop_name]
        num_classes = len(DISEASE_CLASSES[crop_name])
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m = m.to(DEVICE)
        m.eval()
        _disease_models[crop_name] = m
    return _disease_models[crop_name]


# =====================================
# LOAD SENSOR RULES
# =====================================

if os.path.exists(SENSOR_FILE):
    with open(SENSOR_FILE) as f:
        SENSOR_RULES = json.load(f)
else:
    SENSOR_RULES = {}


# =====================================
# PENDING TOKENS  (crop-mismatch confirm)
# =====================================

pending = {}


# =====================================
# HELPER — is this a healthy label?
# =====================================

def _is_healthy(class_name: str) -> bool:
    return class_name.lower().replace(" ", "") in \
           {k.replace(" ", "") for k in HEALTHY_KEYWORDS}


# =====================================
# HELPER — image quality check
# Checks blur, brightness (too dark / too bright), and noise.
# Returns (passed: bool, reason: str)
# reason is a human-readable label shown in the per-image output.
# =====================================

BRIGHTNESS_LOW  = 40    # mean pixel value below this = too dark
BRIGHTNESS_HIGH = 220   # mean pixel value above this = too bright
NOISE_THRESHOLD = 3.5   # std of Laplacian residual above this = noisy

def _check_image_quality(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Unreadable Image"

    # --- Brightness check ---
    mean_val = img.mean()
    if mean_val < BRIGHTNESS_LOW:
        return False, "Too Dark"
    if mean_val > BRIGHTNESS_HIGH:
        return False, "Too Bright"

    # --- Blur check (Laplacian variance) ---
    lap = cv2.Laplacian(img, cv2.CV_64F)
    var = lap.var()
    if var < BLUR_THRESHOLD:
        return False, "Blurred"

    # --- Noise check (std of Laplacian minus its smoothed version) ---
    lap_smooth = cv2.GaussianBlur(lap, (5, 5), 0)
    noise = float(np.std(lap - lap_smooth))
    if noise > NOISE_THRESHOLD * var ** 0.5:
        return False, "Noisy"

    return True, "ok"


# =====================================
# HELPER — Mahalanobis distance
# =====================================

def _mahalanobis(feat):
    if global_mean is None or cov_inv is None:
        return 0.0
    diff = feat - global_mean
    return float(np.sqrt(np.clip(diff @ cov_inv @ diff, 0, None)))


# =====================================
# HELPER — OOD + crop-confidence check
# Returns (passed: bool, detected_crop: str, crop_conf: float, reason: str)
# =====================================

def _ood_check(image_path: str):
    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = crop_classifier(tensor)
            probs   = F.softmax(outputs, dim=1)[0]

        dist      = _mahalanobis(feat_cache["feats"][0])
        top_conf  = probs.max().item() * 100
        top_cls   = CROP_CLASSES[probs.argmax().item()]

        if dist > OOD_THRESHOLD:
            return False, top_cls, top_conf, "OOD_DISTANCE"
        if top_conf < CROP_CONF_MIN:
            return False, top_cls, top_conf, "LOW_CROP_CONF"

        return True, top_cls, top_conf, "ok"

    except Exception as e:
        return False, "", 0.0, f"ERROR:{e}"


# =====================================
# HELPER — top-K inference on one image
# Returns list of (cls, conf%, vote_weight, rank)  or None if low-conf
# =====================================

def _run_topk(image_path: str, model, classes):
    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = F.softmax(outputs, dim=1)

        topk_probs, topk_idx = torch.topk(probs, k=min(TOP_K, len(classes)), dim=1)

        tops = []
        for rank in range(topk_probs.shape[1]):
            cls         = classes[topk_idx[0][rank].item()]
            conf        = topk_probs[0][rank].item() * 100
            vote_weight = conf * (TOP_K_DISCOUNT ** rank)
            tops.append((cls, conf, vote_weight, rank))

            if rank == 0 and conf < PER_IMAGE_MIN_CONFIDENCE:
                return None   # low-confidence — skip vote

        return tops

    except Exception as e:
        return None


# =====================================
# HELPER — sensor rule check
# =====================================

def _check_sensors(crop, disease, sensors):
    if crop not in SENSOR_RULES:
        return True
    if disease not in SENSOR_RULES[crop]:
        return True
    for key, (low, high) in SENSOR_RULES[crop][disease].items():
        if key not in sensors:
            continue
        if not (low <= sensors[key] <= high):
            return False
    return True


# =====================================
# CORE PIPELINE  (used internally)
# =====================================

def _run_full_pipeline(crop_name: str, photos: list, sensors: dict) -> dict:
    """
    Runs the full 4-stage pipeline and returns a result dict.
    crop_name  — the crop name provided by the user
    photos     — list of image file paths
    sensors    — dict with any subset of temp/humidity/uv_index/nitrogen/phosphorous/potassium
    """

    per_image_log = []   # final per-image summary for the response

    # ------------------------------------------------------------------
    # STAGE 1 — Blur Detection
    # ------------------------------------------------------------------

    blur_rejected = []
    not_found     = []
    valid_paths   = []

    for path in photos:
        name = os.path.basename(path)
        if not os.path.exists(path):
            not_found.append(path)
            per_image_log.append({"image": name, "status": "not_found", "prediction": "Image Not Found", "confidence": "--"})
            continue

        passed, reason = _check_image_quality(path)
        if not passed:
            blur_rejected.append(path)
            per_image_log.append({"image": name, "status": "quality_rejected", "prediction": reason, "confidence": "--"})
        else:
            valid_paths.append(path)

    if not valid_paths:
        return {
            "status"    : "error",
            "message"   : "All images were blurry or missing. Retake clearer photos.",
            "images"    : per_image_log
        }

    # ------------------------------------------------------------------
    # STAGE 2 — OOD / Garbage + Unknown-Plant Rejection
    # ------------------------------------------------------------------

    ood_passed   = []
    ood_rejected = {}          # path → reason

    for path in valid_paths:
        passed, det_crop, det_conf, reason = _ood_check(path)
        name = os.path.basename(path)
        if passed:
            ood_passed.append((path, det_crop, det_conf))
        else:
            ood_rejected[path] = reason
            label = "ood_garbage" if reason == "OOD_DISTANCE" else \
                    "unknown_plant" if reason == "LOW_CROP_CONF" else "error"
            per_image_log.append({"image": name, "status": label, "prediction": "Unknown Image", "confidence": "--"})

    if not ood_passed:
        return {
            "status" : "error",
            "message": "No valid crop images found. All images were rejected as non-crop or unknown plant.",
            "images" : per_image_log
        }

    # ------------------------------------------------------------------
    # STAGE 3 — Crop Classification + Weighted Vote
    # ------------------------------------------------------------------

    crop_scores  = defaultdict(float)
    crop_results = []   # (path, tops)

    for path, _pre_crop, _pre_conf in ood_passed:
        tops = _run_topk(path, crop_classifier, CROP_CLASSES)
        if tops is None:
            name = os.path.basename(path)
            per_image_log.append({"image": name, "status": "quality_rejected", "prediction": "Low Quality / Unclear", "confidence": "--"})
            continue
        crop_results.append((path, tops))
        for cls, conf, vote_weight, rank in tops:
            crop_scores[cls] += vote_weight

    if not crop_results:
        return {
            "status" : "error",
            "message": "No valid crop predictions. Images may be too ambiguous.",
            "images" : per_image_log
        }

    total_crop_weight = sum(crop_scores.values())
    detected_crop     = max(crop_scores, key=crop_scores.get)
    crop_conf_pct     = round((crop_scores[detected_crop] / total_crop_weight) * 100, 1)

    # ------------------------------------------------------------------
    # CROP MISMATCH CHECK
    # Returns a warning + token; the front-end /confirm endpoint resumes.
    # ------------------------------------------------------------------

    if crop_name.strip().lower() != detected_crop.lower():
        token = str(uuid.uuid4())
        pending[token] = {
            "crop_name"    : detected_crop,   # use the model-detected crop going forward
            "photos"       : photos,
            "sensors"      : sensors,
        }
        return {
            "status"          : "warning",
            "message"         : (
                f"You selected '{crop_name}' but the classifier detected '{detected_crop}'. "
                f"Do you want to proceed with '{detected_crop}'?"
            ),
            "detected_crop"   : detected_crop,
            "crop_confidence" : f"{crop_conf_pct}%",
            "continue_token"  : token,
            "images"          : per_image_log
        }

    # ------------------------------------------------------------------
    # STAGE 4 — Disease Detection
    # ------------------------------------------------------------------

    disease_model   = _get_disease_model(detected_crop)
    disease_classes = DISEASE_CLASSES[detected_crop]
    disease_results = []   # (path, tops)

    for path, _ in crop_results:
        tops = _run_topk(path, disease_model, disease_classes)
        name = os.path.basename(path)
        if tops is None:
            per_image_log.append({"image": name, "status": "quality_rejected", "prediction": "Low Quality / Unclear", "confidence": "--"})
            continue
        disease_results.append((path, tops))

    if not disease_results:
        return {
            "status" : "error",
            "message": "No valid disease predictions. Retake clearer photos.",
            "images" : per_image_log
        }

    # ------------------------------------------------------------------
    # STAGE 4b — Outlier Rejection
    # Skipped entirely when majority vote is Healthy
    # ------------------------------------------------------------------

    top1_labels      = [tops[0][0] for _, tops in disease_results]

    # Healthy is ONLY the final call if EVERY single valid image voted healthy.
    # Even one disease prediction means the plant could be in early-stage disease —
    # in that case we treat the whole batch as diseased and let the weighted vote
    # decide which disease wins.
    all_healthy      = all(_is_healthy(lbl) for lbl in top1_labels)
    majority_healthy = all_healthy

    # For outlier rejection we only care about the non-healthy majority.
    # Filter out healthy labels first so we find the dominant disease label.
    disease_only_labels = [lbl for lbl in top1_labels if not _is_healthy(lbl)]
    if disease_only_labels:
        majority_disease = Counter(disease_only_labels).most_common(1)[0][0]
    else:
        majority_disease = top1_labels[0]   # all healthy — won't be used

    clean_disease    = []
    outlier_rejected = {}   # path → (cls, conf)

    if majority_healthy:
        # Every image is healthy — keep all, no outlier rejection needed
        clean_disease = list(disease_results)
    else:
        for path, tops in disease_results:
            top_cls, top_conf, _, _ = tops[0]

            # Disease images are NEVER rejected as outliers —
            # a diseased image among many healthy ones is the most important signal
            if not _is_healthy(top_cls):
                clean_disease.append((path, tops))

            # Healthy images: keep them so they contribute their vote weight,
            # but only reject a healthy image as outlier if it is very high-conf
            # AND the dominant label is clearly a disease (not another healthy variant)
            elif top_conf >= OUTLIER_CONF_THRESHOLD and not _is_healthy(majority_disease):
                outlier_rejected[path] = (top_cls, top_conf)

            else:
                clean_disease.append((path, tops))

    # ------------------------------------------------------------------
    # Build per-image log for all surviving images
    # ------------------------------------------------------------------

    clean_paths = {p for p, _ in clean_disease}

    for path, tops in disease_results:
        name     = os.path.basename(path)
        top_cls, top_conf, _, _ = tops[0]
        conf_str = f"{round(top_conf, 1)}%"

        if path in outlier_rejected:
            per_image_log.append({
                "image"      : name,
                "status"     : "outlier_rejected",
                "prediction" : top_cls,
                "confidence" : conf_str
            })
        elif path in clean_paths:
            # sensor check per image
            sensor_ok = _check_sensors(detected_crop, top_cls, sensors)
            per_image_log.append({
                "image"      : name,
                "status"     : "voted" if sensor_ok else "sensor_rejected",
                "prediction" : top_cls,
                "confidence" : conf_str
            })

    # Filter out sensor-rejected from clean_disease
    sensor_passed = []
    for path, tops in clean_disease:
        top_cls = tops[0][0]
        if _check_sensors(detected_crop, top_cls, sensors):
            sensor_passed.append((path, tops))

    if not sensor_passed:
        return {
            "status"         : "error",
            "message"        : "All disease predictions rejected by sensor rules. Check sensor values.",
            "detected_crop"  : detected_crop,
            "crop_confidence": f"{crop_conf_pct}%",
            "images"         : per_image_log
        }

    # ------------------------------------------------------------------
    # FINAL WEIGHTED VOTE
    # ------------------------------------------------------------------

    disease_scores = defaultdict(float)
    for _, tops in sensor_passed:
        for cls, conf, vote_weight, rank in tops:
            disease_scores[cls] += vote_weight

    total_weight    = sum(disease_scores.values())
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: -x[1])

    # Build vote breakdown list (all labels including healthy)
    vote_breakdown = [
        {"disease": d, "vote_share": f"{round((s / total_weight) * 100, 1)}%"}
        for d, s in sorted_diseases
    ]

    # ------------------------------------------------------------------
    # If ANY disease exists in the batch, the final prediction MUST be
    # the top-scoring disease — never healthy — even if healthy images
    # outnumber diseased ones in raw vote weight.
    # ------------------------------------------------------------------

    if not majority_healthy:
        # Pick best disease (skip any healthy labels in the sorted list)
        best_disease   = next((d for d, _ in sorted_diseases if not _is_healthy(d)), None)
        best_weight    = disease_scores[best_disease] if best_disease else 0
        # Compute share against total weight so the % is honest
        weighted_share = round((best_weight / total_weight) * 100, 1) if best_disease else 0

        # Second-best is next non-healthy disease (different from best)
        second_disease = next(
            (d for d, _ in sorted_diseases if not _is_healthy(d) and d != best_disease), None
        )
        second_share   = round((disease_scores[second_disease] / total_weight) * 100, 1) if second_disease else 0
    else:
        best_disease   = sorted_diseases[0][0]
        best_weight    = sorted_diseases[0][1]
        weighted_share = round((best_weight / total_weight) * 100, 1)
        second_disease = sorted_diseases[1][0] if len(sorted_diseases) > 1 else None
        second_share   = round((sorted_diseases[1][1] / total_weight) * 100, 1) if second_disease else 0

    # Uncertain only applies when:
    #   1. All images are healthy but the vote is weak/split
    #   2. Multiple diseases are neck-and-neck (neither clearly leading)
    #
    # If any disease is detected, NEVER suppress it as uncertain just because
    # its vote weight is low — low weight means few images caught it, which is
    # exactly the early-stage signal we want to surface, not hide.
    if not majority_healthy and best_disease is not None:
        uncertain = (
            second_disease is not None and
            abs(weighted_share - second_share) < 5 and
            weighted_share < 30
        )
    else:
        uncertain = (
            weighted_share < WEIGHTED_VOTE_MIN_SHARE or
            best_weight    < MIN_WEIGHT_FOR_DECISION
        )

    # ------------------------------------------------------------------
    # Compose final result
    # ------------------------------------------------------------------

    result = {
        "status"          : "success",
        "detected_crop"   : detected_crop,
        "crop_confidence" : f"{crop_conf_pct}%",
        "vote_breakdown"  : vote_breakdown,
        "images"          : per_image_log,
    }

    if uncertain:
        result["prediction"] = "uncertain"
        result["best_candidate"] = best_disease
        result["weighted_share"] = f"{weighted_share}%"
        result["message"] = (
            f"Prediction is uncertain ({weighted_share}% share, need ≥{WEIGHTED_VOTE_MIN_SHARE}%). "
            "Retake clearer / closer photos and try again."
        )

    elif majority_healthy:
        result["prediction"] = best_disease
        result["weighted_share"] = f"{weighted_share}%"
        result["message"] = "No active disease detected."
        if second_disease and not _is_healthy(second_disease) and second_share > 10:
            result["watch_item"] = {
                "disease"    : second_disease,
                "vote_share" : f"{second_share}%",
                "message"    : "Monitor crop over next 5–7 days."
            }

    else:
        result["prediction"] = best_disease
        result["weighted_share"] = f"{weighted_share}%"
        if second_disease and not _is_healthy(second_disease) and second_share > 15:
            result["also_consider"] = {
                "disease"    : second_disease,
                "vote_share" : f"{second_share}%"
            }

    return result


# =====================================
# PUBLIC API — called by app.py
# =====================================

def run_pipeline(crop_name: str, photos: list, sensors: dict) -> dict:
    """Entry point called from app.py /predict route."""
    sensors = {k: v for k, v in sensors.items() if v not in [None, ""]}
    return _run_full_pipeline(crop_name, photos, sensors)


def resume_pipeline(token: str) -> dict:
    """Called from app.py /confirm route after user confirms crop mismatch."""
    data = pending.pop(token, None)
    if data is None:
        return {"status": "error", "message": "Invalid or expired token."}
    return _run_full_pipeline(
        data["crop_name"],
        data["photos"],
        data["sensors"]
    )