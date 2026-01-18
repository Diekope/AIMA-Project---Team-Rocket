import os
import cv2
from ultralytics import YOLO

# Fonctionnement de base
script_dir = os.path.dirname(os.path.abspath(__file__)) 
model = YOLO(os.path.join(script_dir, "yolo11n.pt"))
root_dir = os.path.dirname(script_dir)
persons_folder = os.path.join(root_dir, "persons")
save_folder = os.path.join(root_dir, "working")

# Pour réaliser une prédiction :
# results = model("persons/a.png", show=True)

def exploit_image(img, img_name, save_folder):
    img_name = os.path.splitext(img_name)[0]
    results = model(img, show=True)

    for result in results:
        # On récupère les bounding boxes
        person_boxes = [b for b in result.boxes if model.names[int(b.cls[0])] == "person"]

        for nn,box in enumerate(person_boxes):
            # On prends les coordonnées pour pouvoir faire la sauvegarde
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            area = result.orig_img[y1:y2, x1:x2]

            save_name = f"{img_name}-person-{nn}-bb-{x1}-{y1}-{x2}-{y2}.png"
            save_path = os.path.join(save_folder, save_name)

            cv2.imwrite(save_path, area)
            print(f"{save_path} sauvegardé")


# Fonction pour parcourir le dossier d'extraction et réaliser les extractions de corps
def run_people_boxes_extraction(persons_folder):
    if os.path.exists(persons_folder):
        os.makedirs(save_folder, exist_ok=True)

        for img in os.listdir(persons_folder):
            print(f"Working with {img}")
            img_path = os.path.join(persons_folder, img)
            exploit_image(img_path, img, save_folder)
    else:
        print(f"Dossier {persons_folder} inexistant")

#run_people_boxes_extraction(persons_folder)