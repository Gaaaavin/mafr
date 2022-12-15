import cv2
import os
from tqdm import tqdm

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
root = "/mnt/NAS/home/xinhao/mafr/data/lfw"
save_root = "/mnt/NAS/home/xinhao/mafr/data/lfw_face"
identities = os.listdir(root)

for identity in tqdm(identities):
    for img_name in os.listdir(os.path.join(root, identity)):
        # Read the input image
        img = cv2.imread(os.path.join(root, identity, img_name))
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = img[x:x+w, y:y+h]
        else:
            face = img
        
        save_path = os.path.join(save_root, identity)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imwrite(os.path.join(save_path, img_name), face)