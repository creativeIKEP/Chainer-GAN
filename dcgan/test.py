import argparse
import numpy as np
import matplotlib.pyplot as plt
import chainer
from net import Generator


def test(n_hidden, npz_path):
    gen = Generator(n_hidden=n_hidden)
    chainer.serializers.load_npz(npz_path, gen)
    noise = np.random.uniform(-1, 1, (1, n_hidden), dtype=np.float32)
    z = Variable(noise)
    x_fake = gen(z, 1)
    print(x_fake)

    im_list = np.asarray(x_fake)
    plt.imshow(im_list)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer-GAN: dcgan')
    parser.add_argument('-n_hidden', '-z', type=int, default=100,
                        help='Size of noise to input Generator. Please input same size when dcgan train')
    parser.add_argument('-path', '-p'
                        help='Trained Generator path. Please input trained generator npz file path')
    args = parser.parse_args()

    test(args.n_hidden, args.path)
