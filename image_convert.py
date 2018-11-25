import numpy as np
import cv2

label_to_pixel_dict = {
    0: [0,0,0],
    1: [128,128,128],
    2: [128,0,0],
    3: [128,64,128],
    4: [0,0,192],
    5: [64,64,128],
    6: [128,128,0],
    7: [192,192,128],
    8: [64, 0, 128],
    9: [192, 128, 128],
    10: [64, 64, 0],
    11: [0, 128,192]
}

pixel_to_label_dict = {tuple(v): k for k, v in label_to_pixel_dict.items()}


def convert_to_label_map(input_image, output_reference, axis = 2):
    for k, v in pixel_to_label_dict.items():
        # RGB to BGR
        set_indexes = (input_image == [k[2], k[1], k[0]]).all(axis=axis)
        output_reference[set_indexes] = v


def convert_bgr_to_rgb(input_image):
    multiplied = (input_image * 255).astype(np.uint8)
    return cv2.cvtColor(multiplied, cv2.COLOR_BGR2RGB)