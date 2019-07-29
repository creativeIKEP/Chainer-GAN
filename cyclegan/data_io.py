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
    #(n, ch, h, w)
    return datasets


def output_images(image_path, real_a_images, fake_b_images, reconstr_a_images, real_b_images, fake_a_images, reconstr_b_images):
    a_image_index = random.randint(0, real_a_images.shape[0] - 1)
    b_image_index = random.randint(0, real_b_images.shape[0] - 1)

    real_a_image = real_a_images[a_image_index]
    fake_b_image = fake_b_images[a_image_index]
    reconstr_a_image = reconstr_a_images[a_image_index]
    real_b_image = real_b_images[b_image_index]
    fake_a_image = fake_a_images[b_image_index]
    reconstr_b_image = reconstr_b_images[b_image_index]

    real_a_image = (real_a_image*128.0+128.0).astype(np.uint8)
    fake_b_image =  (fake_b_image*128.0+128.0).astype(np.uint8)
    reconstr_a_image =  (reconstr_a_image*128.0+128.0).astype(np.uint8)
    real_b_image =  (real_b_image*128.0+128.0).astype(np.uint8)
    fake_a_image =  (fake_a_image*128.0+128.0).astype(np.uint8)
    reconstr_b_image =  (reconstr_b_image*128.0+128.0).astype(np.uint8)

    dst = np.concatenate([real_a_image, fake_b_image], 1)
    dst = np.concatenate([dst, reconstr_a_image], 1)
    dst2 = np.concatenate([real_b_image, fake_a_image], 1)
    dst2 = np.concatenate([dst2, reconstr_b_image], 1)
    result = np.concatenate([dst, dst2], 0)
    Image.fromarray(result).save(image_path + ".png")
