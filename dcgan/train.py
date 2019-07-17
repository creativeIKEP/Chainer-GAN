import data_io
from net import Generator, Discriminator
import numpy as np
import chainer
from chainer import Variable
from chainer.iterators import SerialIterator
import chainer.functions as F
import cupy as xp
from PIL import Image


def train():
    dataset_folder_path = "dataset"
    n_hidden = 300
    epoch_count = 5000
    batch_size = 55

    dataset = data_io.dataset_load(dataset_folder_path)
    train_iter = SerialIterator(dataset, batch_size, repeat=True, shuffle=True)

    gen = Generator(n_hidden=n_hidden)
    dis = Discriminator()

    gen.to_gpu(0)
    dis.to_gpu(0)

    opt_gen = chainer.optimizers.Adam()
    opt_gen.setup(gen)
    opt_dis = chainer.optimizers.Adam()
    opt_dis.setup(dis)

    iteration = 0
    train_iter.reset()
    for epoch in range(epoch_count):
        d_loss_list = []
        g_loss_list = []
        while True:
            mini_batch_images = train_iter.next()

            x_real = Variable(xp.array(mini_batch_images))
            y_real = dis(x_real)

            noise = xp.random.uniform(-1, 1, (batch_size, n_hidden), dtype=np.float32)
            z = Variable(noise)
            x_fake = gen(z, batch_size)
            y_fake = dis(x_fake)

            d_loss = loss_dis(batch_size, y_real, y_fake)
            g_loss = loss_gen(batch_size, y_fake)
            dis.cleargrads()
            gen.cleargrads()
            d_loss.backward()
            g_loss.backward()
            opt_dis.update()
            opt_gen.update()

            d_loss.to_cpu()
            g_loss.to_cpu()

            iteration += batch_size
            d_loss_list.append(d_loss.array)
            g_loss_list.append(g_loss.array)

            if train_iter.is_new_epoch:
                break

        x_fake.to_cpu()
        generated_images = x_fake.array
        generated_images = generated_images.transpose(0, 2, 3, 1)
        Image.fromarray(np.clip(generated_images[0] * 255, 0.0, 255.0).astype(np.uint8)).save(
                   "output/" + str(epoch)+".png")

        print("epoch: " + str(epoch+1) + ", interation: " + str(iteration) + ", d_loss: " + str(np.mean(d_loss_list)) + ", g_loss: " + str(np.mean(g_loss_list)))



def loss_dis(batchsize, y_real, y_fake):
    real_loss = F.softmax_cross_entropy(y_real, Variable(xp.ones(batchsize, dtype=xp.int32)))
    fake_loss = F.softmax_cross_entropy(y_fake, Variable(xp.zeros(batchsize, dtype=xp.int32)))
    return real_loss + fake_loss


def loss_gen(batchsize, y_fake):
    fake_loss = F.softmax_cross_entropy(y_fake, Variable(xp.ones(batchsize, dtype=xp.int32)))
    return fake_loss


if __name__ == '__main__':
    train()
