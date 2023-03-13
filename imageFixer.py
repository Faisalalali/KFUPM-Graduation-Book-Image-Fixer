from tqdm import tqdm
import cv2
import os

import numpy as np

INPUT_DIR = r"images"
OUTPUT_DIR = r"output"

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to align and crop an image


def create_canvas():
    # Create a canvas
    canvas = np.zeros((2250, 1500, 3), dtype=np.uint8)

    # Fill the canvas with gradient white to light gray from center to sides
    Y = np.linspace(-1, 1, 2250)
    X = np.linspace(-1, 1, 1500)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(X**2 + Y**2)
    Z = Z / np.max(Z)
    Z = Z[:, :, np.newaxis]

    canvas = (1 - Z) * (227, 227, 227) + Z * (186, 186, 186)

    return canvas


def select_largest_face(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    # If no faces are detected, try again with lower accuracy
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Select the largest face
    max_area = 0
    max_face = None
    for face in faces:
        x, y, w, h = face
        area = w * h
        if area > max_area:
            max_area = area
            max_face = face

    return max_face


def align(image):
    # Create a canvas
    canvas = create_canvas()

    # Select the largest face
    max_face = select_largest_face(image)

    if max_face is None:
        return None

    x, y, w, h = max_face

    center_x = x + w//2
    center_y = y + h//2 + 666

    # Calculate the center of the canvas
    canvas_center_x = canvas.shape[1] // 2
    canvas_center_y = canvas.shape[0] // 2

    # Calculate the shift needed to center the face on the canvas
    shift_x = canvas_center_x - center_x
    shift_y = canvas_center_y - center_y

    # Shift the image and create an alpha mask
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, M, (canvas.shape[1], canvas.shape[0]))

    # Define the mask of missing pixels
    mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
    mask = cv2.rectangle(
        mask, (0, 0), (canvas.shape[1], canvas.shape[0]), (255, 255, 255), -1)
    mask = cv2.warpAffine(mask, M, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.bitwise_not(mask)

    # Apply the inpainting algorithm to fill the missing pixels
    filled = cv2.inpaint(shifted, mask, 3, cv2.INPAINT_TELEA)

    return filled


def zoom(image):
    # zoom to 1.25x
    zoomed = cv2.resize(image, (0, 0), fx=1.25, fy=1.25)

    return zoomed


# Loop over all images and align/crop them
total_count = len(os.listdir(INPUT_DIR))
print(f"Total number of images: {total_count}")
for filename in tqdm(os.listdir(INPUT_DIR)):
    image = cv2.imread(os.path.join(INPUT_DIR, filename))
    try:
        aligned = zoom(align(image))
    except:
        aligned = image
    if aligned is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), aligned)
print(f"Number of aligned images: {len(os.listdir(OUTPUT_DIR))}")
