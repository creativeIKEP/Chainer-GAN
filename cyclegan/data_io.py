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


def output_images(real_a_images, fake_b_images, reconstr_a_images, real_b_images, fake_a_images, reconstr_b_images):
    Image.fromarray(np.clip(generated_images[0] * 255, 0.0, 255.0).astype(np.uint8)).save(image_path + str(epoch)+".png")
