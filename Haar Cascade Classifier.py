import cv2

def detect_objects_haar(image_path, cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return objects

# Example usage:
image_path = "image.jpg"
cascade_path = "haarcascade_frontalface_default.xml"
objects = detect_objects_haar(image_path, cascade_path)
print("Detected objects:", objects)
