import cv2

# Colors for different classes
CLASS_COLORS = {
    "Myringosclerosis": (255, 0, 0),  # Blue
    "OtitisMedia": (0, 255, 0),       # Green
    "Normal": (0, 0, 255),            # Red
}

def draw_boxes(image, predictions):
    """
    Draw bounding boxes and labels on OpenCV image.
    """
    for pred in predictions:
        x, y, w, h = pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")
        cls = pred.get("class")
        conf = pred.get("confidence", 0)
        if None not in (x, y, w, h):
            color = CLASS_COLORS.get(cls, (255, 255, 0))
            start_point, end_point = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(image, start_point, end_point, color, 2)
            cv2.putText(image, f"{cls} ({conf*100:.1f}%)", (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image
