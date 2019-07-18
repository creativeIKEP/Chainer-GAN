import data_io
import logGraph
from net import Generator, Discriminator
import json
import datetime
import os
from PIL import Image
import numpy as np
import cupy as xp
import chainer
from chainer.serializers import save_npz
from chainer import Variable
from chainer.iterators import SerialIterator
import chainer.functions as F


def train():
    dataset_folder_path = "dataset"
    n_hidden = 100
    epoch_count = 3#4294967295
    batch_size = 55

    dataset = data_io.dataset_load(dataset_folder_path)
    train_iter = SerialIterator(dataset, batch_size, repeat=True, shuffle=True)

    gen = Generator(n_hidden=n_hidden)
    dis = Discriminator()

    gen.to_gpu(0)
    dis.to_gpu(0)

    opt_gen = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_gen.setup(gen)
    opt_dis = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_dis.setup(dis)

    iteration = 0
    train_iter.reset()

    log_list = []
    now_time = datetime.datetime.now()
    folder_name = "{0:%Y-%m-%d_%H-%M}".format(now_time)
    output_path = "output/" + folder_name + "/"
    image_path = output_path + "image/"
    dis_model_path = output_path + "dis/"
    gen_model_path = output_path + "gen/"
    os.mkdir(output_path)
    os.mkdir(image_path)
    os.mkdir(dis_model_path)
    os.mkdir(gen_model_path)

    for epoch in range(epoch_count):
        d_loss_list = []
        g_loss_list = []
        while True:
            mini_batch_images = train_iter.next()
            mini_batch_images = np.array(mini_batch_images)
            mini_batch_images = (mini_batch_images - 128.0) / 128.0
            x_real = Variable(np.array(mini_batch_images))

            x_real.to_gpu(0)
            y_real = dis(x_real)

            noise = xp.random.uniform(-1, 1, (batch_size, n_hidden), dtype=np.float32)
            z = Variable(noise)
            x_fake = gen(z, batch_size)
            y_fake = dis(x_fake)

            d_loss = loss_dis(batch_size, y_real, y_fake)
            g_loss = loss_gen(batch_size, y_fake)

            gen.cleargrads()
            g_loss.backward()
            opt_gen.update()

            dis.cleargrads()
            d_loss.backward()
            opt_dis.update()

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
                   image_path + str(epoch)+".png")

        print("epoch: " + str(epoch) + ", interation: " + str(iteration) + ", d_loss: " + str(np.mean(d_loss_list)) + ", g_loss: " + str(np.mean(g_loss_list)))

        log_json = {"epoch": str(epoch), "iteration": str(iteration), "d_loss": str(np.mean(d_loss_list)), "g_loss": str(np.mean(g_loss_list))}
        log_list.append(log_json)
        with open(output_path + 'log.json', 'w') as log_file:
            json.dump(log_list, log_file, indent=4)

        if(epoch % 100 == 0):
            dis.to_cpu()
            save_npz(dis_model_path + str(epoch) + '.npz', dis)
            gen.to_cpu()
            save_npz(gen_model_path + str(epoch) + '.npz', gen)
            gen.to_gpu(0)
            dis.to_gpu(0)

    logGraph.save_log_graph(output_path + 'log.json', output_path + "lossGraph.png")
    dis.to_cpu()
    save_npz(dis_model_path + 'last.npz', dis)
    gen.to_cpu()
    save_npz(gen_model_path + 'last.npz', gen)


def loss_dis(batchsize, y_real, y_fake):
    real_loss = F.softmax_cross_entropy(y_real, Variable(xp.ones(batchsize, dtype=xp.int32)))
    fake_loss = F.softmax_cross_entropy(y_fake, Variable(xp.zeros(batchsize, dtype=xp.int32)))
    return real_loss + fake_loss


def loss_gen(batchsize, y_fake):
    fake_loss = F.softmax_cross_entropy(y_fake, Variable(xp.ones(batchsize, dtype=xp.int32)))
    return fake_loss


if __name__ == '__main__':
    train()
