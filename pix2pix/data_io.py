import os
import glob
import numpy as np
from PIL import Image
import random
import chainer


def dataset_load(dataset_folder_path):
    dataset_path = os.path.join(dataset_folder_path, '*')
    image_files_path_list = glob.glob(dataset_path)
    datasets = chainer.datasets.ImageDataset(image_files_path_list)
    datasets = np.asarray(datasets, dtype=np.float32)

    width = 256
    height = width * 9 //16 #change full HD aspect
    datasets = chainer.functions.resize_images(datasets, (height, width))
    datasets = (datasets - 128.0) / 128.0

    #(n, ch, h, w)
    return datasets.array


def output_images(image_path, input_images, correct_images, generate_images):
    image_index = random.randint(0, input_images.shape[0] - 1)

    input_image = input_images[image_index]
    correct_image = correct_images[image_index]
    generate_image = generate_images[image_index]

    input_image = (input_image*128.0+128.0).astype(np.uint8)
    correct_image = (correct_image*128.0+128.0).astype(np.uint8)
    generate_image = (generate_image*128.0+128.0).astype(np.uint8)

    dst = np.concatenate([input_image, correct_image], 1)
    result = np.concatenate([dst, generate_image], 1)
    Image.fromarray(result).save(image_path + ".png")
