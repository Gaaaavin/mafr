import os
import dlib
from MaskTheFace import mask_image, download_dlib_model


def init():
    # Set up dlib face detector and predictor
    detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "utils/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()

    return detector


def add_mask(image_path, args):
    masked_image, _, _, original_image = mask_image(image_path, args)
    return masked_image[0], original_image