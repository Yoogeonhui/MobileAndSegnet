import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import image_convert


class DataLoader:

    def __init__(self, rgb_image_path, ground_truth_path, start_batch = 0, batch_size = 16, width = 16*16, height = 16*9):
        self.rgb_file_names = [rgb_image_path + f for f in listdir(rgb_image_path) if isfile(join(rgb_image_path, f))]
        print(self.rgb_file_names)
        self.ground_file_names = [ground_truth_path+f for f in listdir(rgb_image_path) if isfile(join(rgb_image_path, f))]
        print(self.ground_file_names)
        self.batch_size = batch_size
        self.batch_num = start_batch
        self.batch_max = len(self.rgb_file_names)//self.batch_size
        self.width = width
        self.height = height


    def get_next_batch(self):
        next_epoch = False
        start = self.batch_num * self.batch_size
        end = (self.batch_num+1) * self.batch_size
        if end >= self.batch_max * self.batch_size:
            start = 0
            end = self.batch_size
            next_epoch = True
            self.batch_num = 0
        output_array_image = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        output_array_ground = np.zeros((self.batch_size, self.height, self.width), dtype=np.int8)
        cnt = 0
        for i in range(start, end):
            rgb_image = cv2.imread(self.rgb_file_names[i])
            rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
            rgb_image = rgb_image.astype(np.float64)
            ground_image = cv2.imread(self.ground_file_names[i])
            ground_image = cv2.resize(ground_image, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
            rgb_image /= 255
            output_array_image[cnt, :, :, :] = rgb_image
            image_convert.convert_to_label_map(ground_image, output_array_ground[cnt, :, :])
            cnt += 1
        self.batch_num += 1

        return next_epoch, output_array_image, output_array_ground

