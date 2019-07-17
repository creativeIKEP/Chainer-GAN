import data_io
from net import Generator, Discriminator
import numpy as np
import chainer
from chainer import Variable
from chainer.iterators import SerialIterator
import chainer.functions as F


def train():
    dataset_folder_path = "dataset"
    n_hidden = 300
    epoch_count = 50
    batch_size = 1

    dataset = data_io.dataset_load(dataset_folder_path)
    train_iter = SerialIterator(dataset, batch_size, repeat=True, shuffle=True)

    gen = Generator(n_hidden=n_hidden)
    dis = Discriminator()
    #gen.to_gpu(0)
    #dis.to_gpu(0)

    opt_gen = chainer.optimizers.Adam()
    opt_gen.setup(gen)
    opt_dis = chainer.optimizers.Adam()
    opt_dis.setup(dis)

    iteration = 0
    for epoch in range(epoch_count):
        d_loss_list = []
        g_loss_list = []
        while True:
            mini_batch_images = train_iter.next()

            x_real = Variable(np.array(mini_batch_images))
            y_real = dis(x_real)

            noise = np.random.uniform(-1, 1, (batch_size, n_hidden)).astype(np.float32)
            z = Variable(noise)
            x_fake = gen(z, (batch_size))
            y_fake = dis(x_fake)

            dis.cleargrads()
            gen.cleargrads()
            d_loss = loss_dis(batch_size, y_real, y_fake)
            g_loss = loss_gen(batch_size, y_fake)
            d_loss.backward()
            g_loss.backward()
            opt_dis.update()
            opt_gen.update()

            iteration += batch_size
            d_loss_list.append(d_loss.array)
            g_loss_list.append(g_loss.array)

            if train_iter.is_new_epoch:
                break

        print("epoch: " + str(epoch+1) + ", interation: " + str(iteration) + ", d_loss: " + str(np.mean(d_loss_list)) + ", g_loss: " + str(np.mean(g_loss_list)))



def loss_dis(batchsize, y_real, y_fake):
    real_loss = F.softmax_cross_entropy(y_real, Variable(np.ones(batchsize, dtype=np.int32)))
    fake_loss = F.softmax_cross_entropy(y_fake, Variable(np.zeros(batchsize, dtype=np.int32)))
    L1 = F.sum(real_loss) / batchsize
    L2 = F.sum(fake_loss) / batchsize
    return L1 + L2


def loss_gen(batchsize, y_fake):
    fake_loss = F.softmax_cross_entropy(y_fake, Variable(np.ones(batchsize, dtype=np.int32)))
    loss = F.sum(fake_loss) / batchsize
    return loss


if __name__ == '__main__':
    train()
