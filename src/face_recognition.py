import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions

model = VGGFace(model="vgg16", include_top=True)

def predict_celebrity(face_path):
    img = cv2.imread(face_path)
    if img is None:
        raise FileNotFoundError(face_path)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img.astype(np.float32), axis=0)
    x = preprocess_input(x, version=1)  
    preds = model.predict(x)
    decoded = decode_predictions(preds)[0]
    name,_ = decoded[0]
    return name

def main():
    print(predict_celebrity("./visage.jpg"))
main()