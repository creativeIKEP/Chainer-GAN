import data_io
from net import Generator, Discriminator
import argparse
import json
import datetime
import os
import numpy as np
import cupy as xp
import chainer
from chainer.serializers import save_npz
from chainer import Variable
from chainer.iterators import SerialIterator
import chainer.functions as F


def train(batch_size, epoch_count, lamda, datasetA_folder_path, datasetB_folder_path, output_path):
    print("Start load images data from " + datasetA_folder_path)
    dataset_A = data_io.dataset_load(datasetA_folder_path)
    print("Finish load images data from " + datasetA_folder_path)
    print("Start load images data from " + datasetB_folder_path)
    dataset_B = data_io.dataset_load(datasetB_folder_path)
    print("Finish load images data from " + datasetB_folder_path)

    if len(dataset_A) != len(dataset_B):
        print("Error! Datasets are not paired data.")
        exit()

    gen = Generator()
    dis = Discriminator()

    gen.to_gpu(0)
    dis.to_gpu(0)

    opt_g = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_g.setup(gen)
    opt_d = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_d.setup(dis)

    iteration = 0

    log_list = []
    image_path = output_path + "image/"
    dis_model_path = output_path + "dis/"
    gen_model_path = output_path + "gen/"
    os.mkdir(output_path)
    os.mkdir(image_path)
    os.mkdir(dis_model_path)
    os.mkdir(gen_model_path)

    dataset_num = min(dataset_A.shape[0], dataset_B.shape[0])
    for epoch in range(epoch_count):
        d_loss_list = []
        g_loss_list = []
        for i in range(dataset_num//batch_size):
            input_image = dataset_A[i*batch_size:(i+1)*batch_size]
            input_image = Variable(input_image)
            input_image.to_gpu(0)

            correct_image = dataset_B[i*batch_size:(i+1)*batch_size]
            correct_image = Variable(correct_image)
            correct_image.to_gpu(0)

            fake_image = gen(input_image)

            d_real_result = dis(correct_image)
            d_fake_result = dis(fake_image)
            loss_d = loss_dis(batch_size, d_real_result, d_fake_result)

            dis.cleargrads()
            loss_d.backward()
            opt_d.update()


            """generatorのloss計算"""
            loss_g = loss_gen(d_fake_result, fake_image, correct_image, lamda)

            gen.cleargrads()
            loss_g.backward()
            opt_g.update()

            loss_d.to_cpu()
            loss_g.to_cpu()

            iteration += batch_size
            d_loss_list.append(loss_d.array)
            g_loss_list.append(loss_g.array)

        input_image.to_cpu()
        correct_image.to_cpu()
        fake_image.to_cpu()
        input_images = input_image.array.transpose(0, 2, 3, 1)
        correct_images = correct_image.array.transpose(0, 2, 3, 1)
        fake_images = fake_image.array.transpose(0, 2, 3, 1)
        data_io.output_images(image_path + str(epoch), input_images, correct_images, fake_images)

        print("epoch: " + str(epoch) + ", interation: " + str(iteration) + \
            ", d_loss: " + str(np.mean(d_loss_list)) + ", g_loss: " + str(np.mean(g_loss_list)))

        log_json = {"epoch": str(epoch), "interation": str(iteration), \
            "d_loss": str(np.mean(d_loss_list)),"g_BA_loss": str(np.mean(g_loss_list))}
        log_list.append(log_json)
        with open(output_path + 'log.json', 'w') as log_file:
            json.dump(log_list, log_file, indent=4)

        if(epoch % 100 == 0):
            gen.to_cpu()
            dis.to_cpu()
            save_npz(gen_model_path + str(epoch) + '.npz', gen)
            save_npz(dis_model_path + str(epoch) + '.npz', dis)
            gen.to_gpu(0)
            dis.to_gpu(0)


    gen.to_cpu()
    dis.to_cpu()
    save_npz(gen_model_path + 'last.npz', gen)
    save_npz(dis_model_path + 'last.npz', dis)


def loss_dis(batchsize, real_result, fake_result):
    batchsize, ch, w, h = real_result.data.shape
    real_loss = F.mean_squared_error(real_result, Variable(xp.ones((batchsize, ch, w, h), dtype=xp.float32)))
    fake_loss = F.mean_squared_error(fake_result, Variable(xp.zeros((batchsize, ch, w, h), dtype=xp.float32)))
    return real_loss + fake_loss


def loss_gen(d_fake_result, fake_image, correct_image, lamda):
    batchsize, ch, w, h = d_fake_result.data.shape
    adversarial_loss = F.mean_squared_error(d_fake_result, Variable(xp.ones((batchsize, ch, w, h), dtype=xp.float32)))
    consistency_loss = F.mean_absolute_error(correct_image, fake_image)
    return adversarial_loss + lamda * consistency_loss


if __name__ == '__main__':
    now_time = datetime.datetime.now()
    folder_name = "{0:%Y-%m-%d_%H-%M-%S}".format(now_time)

    parser = argparse.ArgumentParser(description='Chainer-GAN: pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--lamda', '-l', type=float, default=10.0,
                        help='Percentage of loss caluculation between "Consistency Loss" and "Adversarial Loss"')
    parser.add_argument('--datasetA', '-dA', default='dataset/datasetA',
                        help='Directory of dataset image files. Default is /Chainer-GAN/pix2pix/dataset/datasetA')
    parser.add_argument('--datasetB', '-dB', default='dataset/datasetB',
                        help='Directory of dataset image files. Default is /Chainer-GAN/pix2pix/dataset/datasetB')
    parser.add_argument('--out', '-o', default='output/' + folder_name + "/",
                        help='Directory to output the result. Default is /Chainer-GAN/pix2pix/output/yyyy-mm-dd_HH-MM-SS')
    args = parser.parse_args()
    train(args.batchsize, args.epoch, args.lamda, args.datasetA, args.datasetB, args.out)
