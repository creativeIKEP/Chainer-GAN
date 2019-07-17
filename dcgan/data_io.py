import os
import glob
import numpy as np
from PIL import Image
import chainer


def dataset_load(dataset_folder_path):
    dataset_path = os.path.join(dataset_folder_path, '*')
    image_files_path_list = glob.glob(dataset_path)
    datasets = chainer.datasets.ImageDataset(image_files_path_list)
    #(n, ch, h, w) 
    return datasets


def reshape_image(image_path, width, height):
    img = Image.open(image_path)
    img_resize = img.resize((width, height))
    img_resize.save(image_path)


def combine_images(generated_images):
    image_count = generated_images.shape[0]
    width_count = int(math.sqrt(image_count))
    height_count = int(math.ceil(float(image_count) / width_count))
    width_shape = generated_images.shape[1]
    height_shape = generated_images.shape[2]
    combine_image = np.zeros((height_shape*height_count, width_shape*width_count, 3), dtype=generated_images.dtype)

    index_count=0
    for j in range(height_count):
        for i in range(width_count):
            if image_count<=index_count:
                return combine_image
            combine_image[j*height_shape:(j+1)*height_shape, i*width_shape:(i+1)*width_shape, :] = \
                generated_images[index_count]
            index_count += 1

    return combine_image
