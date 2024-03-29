import argparse
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import Variable
from net import Generator


def test(n_hidden, npz_path):
    with chainer.using_config("train", False):
        gen = Generator(n_hidden=n_hidden)
        chainer.serializers.load_npz(npz_path, gen)
        noise = np.random.uniform(-1, 1, (1, n_hidden)).astype(np.float32)
        z = Variable(noise)
        x_fake = gen(z, 1)
        generated_image = x_fake.array
        generated_image = generated_image.transpose(0, 2, 3, 1)
        generated_image = np.clip(generated_image[0] * 255, 0.0, 255.0).astype(np.uint8)

        im_list = np.asarray(generated_image)
        plt.imshow(im_list)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer-GAN: dcgan')
    parser.add_argument('-n_hidden', '-z', type=int, default=100,
                        help='Size of noise to input Generator. Please input same size when dcgan train')
    parser.add_argument('-path', '-p', default=None,
                        help='Trained Generator path. Please input trained generator npz file path')
    args = parser.parse_args()
    if args.path is None:
        print("Please input trained generator npz file path.")
        print("$ python test.py -path 'npz_file_path'")
        exit()

    test(args.n_hidden, args.path)
