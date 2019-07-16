import data_io
from net import Generator, Discriminator
import chainer
from chainer import Variable
import numpy as np


def train():
    dataset = data_io.dataset_load("dataset")

    gen = Generator(n_hidden=300)
    dis = Discriminator()
    #gen.to_gpu(0)
    #dis.to_gpu(0)

    opt_gen = chainer.optimizers.Adam()
    opt_gen.setup(gen)
    opt_dis = chainer.optimizers.Adam()
    opt_dis.setup(dis)

    #dtype = chainer.get_dtype()
    image = gen(Variable(np.ones((1, 300), dtype=np.float32)), 1)

    a = dis(image)
    print(a)

if __name__ == '__main__':
    train()
