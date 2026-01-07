"""
        _summary_: Extract faces from images using RetinaFace model.
"""
from retinaface import RetinaFace
from PIL import Image
import numpy as np
import os

def extract_faces(image_path, filename):
    """
    Extract faces from one single image and save them in the save_folder
    """
    faces = RetinaFace.detect_faces(image_path)
    real_faces = {}
    img = Image.open(image_path) # Load image with PIL
    img = img.convert("RGB")
    img = np.array(img) # Convert to array for processing
    height, width, channels = img.shape
    
    # Create a sub_folder for each image to store the extracted faces
    sub_folder_path = os.path.join(save_folder, os.path.splitext(filename)[0])
    os.makedirs(sub_folder_path, exist_ok=True)
    
    """
    Step 1 : Data cleansing
    """
    # We check that faces is a dictionary (in case of no face detected)
    if isinstance(faces, dict):
        for id, values in faces.items():
            if "score" in values:
                # We filter the false positive with a threshold of 0.9 for face detection
                if values["score"] > 0.9:
                    real_faces[id] = values
        
        # We check that the rectangle stays within the image boundaries
        for id, values in real_faces.items():
            facial_area = values["facial_area"]
            # x1 = x top left
            # y1 = y top left
            # x2 = x botton right
            # y2 = y botton right
            x1, y1, x2, y2 = facial_area
            
            # Add 15% padding around the face for better extraction and lisibility
            w = int(x2 - x1)
            h = int(y2 - y1)
            x1 = x1 - 0.15 * w
            y1 = y1 - 0.15 * h
            x2 = x2 + 0.15 * w
            y2 = y2 + 0.15 * h

            # Ensure that the coordinates stays within the image boundaries
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width, x2))
            y2 = int(min(height, y2))
            """
            Step 2 : Extraction faces and save them in the save_folder
            """
            face = img[y1:y2, x1:x2] # Extract face from the image
            Image.fromarray(face).save(os.path.join(sub_folder_path,os.path.splitext(filename)[0]+f"_face_{id}.png"))


def extract_all_faces_in_folder(persons_folder):
    if not os.path.exists(save_folder):
        # If the folder doesn't exist, we create it
        os.makedirs(save_folder, exist_ok=True)
    
    # Loop on all the images in the folder persons_folder
    for img in os.listdir(persons_folder):
        if not os.path.exists(save_folder):
            print(f"Error: Folder {save_folder} does not exist.")
            break
        
        os.makedirs(save_folder, exist_ok=True)
        img_path = os.path.join(persons_folder, img)
        extract_faces(img_path, img)


# -------------------------- Main Program --------------------------------
save_folder = "faces"
extract_folder = "persons"

