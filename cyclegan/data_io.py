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


def output_images(image_path, real_a_images, fake_b_images, reconstr_a_images, real_b_images, fake_a_images, reconstr_b_images):
    a_image_index = random.randint(0, real_a_images.shape[0] - 1)
    b_image_index = random.randint(0, real_b_images.shape[0] - 1)

    real_a_image = real_a_images[a_image_index]
    fake_b_image = fake_b_images[a_image_index]
    reconstr_a_image = reconstr_a_images[a_image_index]
    real_b_image = real_b_images[b_image_index]
    fake_a_image = fake_a_images[b_image_index]
    reconstr_b_image = reconstr_b_images[b_image_index]

    real_a_image = np.clip(real_a_image * 255, 0.0, 255.0).astype(np.uint8)
    fake_b_image = np.clip(fake_b_image * 255, 0.0, 255.0).astype(np.uint8)
    reconstr_a_image = np.clip(reconstr_a_image * 255, 0.0, 255.0).astype(np.uint8)
    real_b_image = np.clip(real_b_image * 255, 0.0, 255.0).astype(np.uint8)
    fake_a_image = np.clip(fake_a_image * 255, 0.0, 255.0).astype(np.uint8)
    reconstr_b_image = np.clip(reconstr_b_image * 255, 0.0, 255.0).astype(np.uint8)

    a_w = real_a_image.shape[2]
    a_h = real_a_image.shape[1]
    b_w = real_b_image.shape[2]
    b_h = real_b_image.shape[1]

    dst = Image.new('RGB', (max(a_w*3, b_w*3), a_h + b_h), (1, 1, 1))
    dst.paste(real_a_image, (0, 0))
    dst.paste(fake_b_image, (a_w, 0))
    dst.paste(reconstr_a_image, (a_w*2, 0))
    dst.paste(real_b_image, (0, a_h))
    dst.paste(fake_a_image, (b_w, a_h))
    dst.paste(reconstr_b_image, (b_w*2, a_h))
    dst.save(image_path + ".png")

    Image.fromarray(np.clip(generated_images[0] * 255, 0.0, 255.0).astype(np.uint8)).save(image_path + str(epoch)+".png")
