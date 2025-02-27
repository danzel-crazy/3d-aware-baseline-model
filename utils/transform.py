import cv2
import numpy as np

def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h
    return None

def map_mask(mask1, mask2):
    bbox1 = get_bounding_box(mask1)
    bbox2 = get_bounding_box(mask2)

    if bbox1 is None or bbox2 is None:
        raise ValueError("One of the masks has no detected object.")

    x1, y1, w1, h1 = bbox1  # Mask 1 location & size
    x2, y2, w2, h2 = bbox2  # Mask 2 location & size
    scale_x = w1 / w2
    scale_y = h1 / h2

    mask2_resized = cv2.resize(mask2, (int(w2 * scale_y), int(h2 * scale_y)), interpolation=cv2.INTER_NEAREST)
    aligned_mask2 = np.zeros_like(mask1)

    # Compute new top-left coordinates to align mask2 with mask1
    new_x = x1
    new_y = y1
    aligned_mask2[new_y:new_y+mask2_resized.shape[0], new_x:new_x+mask2_resized.shape[1]] = mask2_resized

    return aligned_mask2

def map_object_to_mask(mask1, object_img, object_mask):
    """Align and scale the object to match the position and size of mask1."""
    bbox1 = get_bounding_box(mask1)
    bbox2 = get_bounding_box(object_mask)

    if bbox1 is None or bbox2 is None:
        raise ValueError("One of the masks has no detected object.")

    # Extract bounding box properties
    x1, y1, w1, h1 = bbox1  # Mask 1 location & size
    x2, y2, w2, h2 = bbox2  # Object location & size

    # Compute scale factors
    scale_x = w1 / w2
    scale_y = h1 / h2

    # Resize object image to fit Mask 1's size
    object_resized = cv2.resize(object_img, (int(w2 * scale_y), int(h2 * scale_y)), interpolation=cv2.INTER_LINEAR)

    # Create an empty canvas for transformed object
    aligned_object = np.zeros_like(mask1)

    # Convert to 3-channel (if needed)
    if len(aligned_object.shape) == 2:
        aligned_object = cv2.cvtColor(aligned_object, cv2.COLOR_GRAY2BGR)

    # Compute new top-left coordinates to align object with mask1
    new_x, new_y = x1, y1

    # Place resized object onto the empty canvas
    h, w, _ = object_resized.shape
    aligned_object[new_y:new_y+h, new_x:new_x+w] = object_resized

    return aligned_object


def mask_transform(mask1, mask2):
    # Ensure masks are binary (0 or 255)
    mask1 = cv2.threshold(mask1, 128, 255, cv2.THRESH_BINARY)[1]
    # mask2 = cv2.threshold(mask2, 128, 255, cv2.THRESH_BINARY)[1]

    # Transform mask2 to match mask1
    aligned_mask2 = map_mask(mask1, mask2)

    return aligned_mask2

def object_transform(mask1, object_img, object_mask):
    # Ensure masks are binary (0 or 255)
    mask1 = cv2.threshold(mask1, 128, 255, cv2.THRESH_BINARY)[1]
    object_mask = cv2.threshold(object_mask, 128, 255, cv2.THRESH_BINARY)[1]

    # Transform object to match mask1
    aligned_object = map_object_to_mask(mask1, object_img, object_mask)

    return aligned_object

def compute_brightness(image):
    """Computes mean brightness of non-black pixels in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    non_black_pixels = gray > 0  # Mask of non-black pixels
    if np.count_nonzero(non_black_pixels) == 0:
        return 0  # Avoid division by zero if object is fully black
    return np.mean(gray[non_black_pixels])  # Compute mean brightness of object only

def match_brightness(src, target):
    """Adjusts the brightness of the target image to match the source image."""
    src_brightness = compute_brightness(src)
    target_brightness = compute_brightness(target)

    if target_brightness == 0:  # Avoid division by zero
        return target

    brightness_ratio = src_brightness / target_brightness
    adjusted_target = cv2.convertScaleAbs(target, alpha=brightness_ratio, beta=0)  # Scale brightness
    return adjusted_target
