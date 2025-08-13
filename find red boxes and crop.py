import cv2
import numpy as np

def find_red_boxes_and_crops(image_bgr,
                             min_area=500,
                             pad=4,
                             pick='largest'
                             ):
    """
    Detect red rectangles/regions in an image and return:
      - boxes: list of (x, y, w, h)
      - crops: list of cropped BGR images (with padding & clamped to image bounds)
      - vis:   visualization image with boxes drawn
    image_bgr: OpenCV BGR image
    """
    H, W = image_bgr.shape[:2]

    # 1) Convert to HSV (OpenCV: H in [0,179])
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 2) Red color mask (two ranges because red wraps around hue)
    # lower reds (0-10)
    lower1 = np.array([0, 160, 20],   dtype=np.uint8)
    upper1 = np.array([10,255,255],   dtype=np.uint8)
    # upper reds (160-180)
    lower2 = np.array([160,160,20],   dtype=np.uint8)
    upper2 = np.array([180,255,255],  dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # 3) Morphology to clean noise & close gaps
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 4) Find external contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) Filter by area and (optionally) aspect ratio / rectangularity
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        candidates.append((x, y, w, h, area))

    if not candidates:
        return [], [], image_bgr.copy()  # nothing found

    # 6) Choose boxes
    if pick == 'largest':
        candidates.sort(key=lambda t: t[4], reverse=True)
        candidates = [candidates[0]]  # keep only the largest

    # 7) Build crops with padding and clamp to image borders
    boxes = []
    crops = []
    vis = image_bgr.copy()
    for (x, y, w, h, _) in candidates:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        crop = image_bgr[y0:y1, x0:x1].copy()
        boxes.append((x0, y0, x1-x0, y1-y0))
        crops.append(crop)

        # draw nice rectangle
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0,255,0), 2)

    return boxes, crops, vis
