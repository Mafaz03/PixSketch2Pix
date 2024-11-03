import numpy as np
from utils import image_to_line_art
import cv2
import config
import os
from PIL import Image
import matplotlib.pyplot as plt

canvas = np.zeros((256, 256*3, 3), dtype=np.uint8)

standard_size = config.IMAGE_SIZE

folders = ["image_dataset/drones/train", "image_dataset/drones/val"]

orignal_size = 600

for folder in folders:
    files = os.listdir(folder)
    for file in files:
        image_pth = os.path.join(folder, file)
        input = np.array(Image.open(image_pth))[:, :orignal_size, ...]
        target = np.array(Image.open(image_pth))[:, orignal_size:, ...]


        input_resized = cv2.resize(input, (standard_size, standard_size))
        line_art_resized = cv2.resize(image_to_line_art(input), (standard_size, standard_size))
        target_resized = cv2.resize(target, (standard_size, standard_size))

        canvas[:, :standard_size, ...] = input_resized
        canvas[:, standard_size:standard_size*2, ...] = line_art_resized
        canvas[:, standard_size*2:, ...] = target_resized

        plt.imsave(image_pth, canvas)