import sys
import os

# Ajouter la racine du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
working_folder_ = os.path.join(project_root, "img/working")

# print(working_folder)


if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Maintenant l'import absolu fonctionnera
from pre.bounding import exploit_image

def split_bodies_and_save_faces(img, img_name, working_folder=working_folder_):
    working_folder = exploit_image(img, img_name, working_folder)

    if working_folder:
        people_folder = os.path.join(working_folder_, working_folder)
        print(people_folder)