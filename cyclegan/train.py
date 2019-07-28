import data_io
from net import Generator, Discriminator
import chainer


def train(batch_size, epoch_count, datasetA_folder_path, datasetB_folder_path, output_path):
    dataset_A = data_io.dataset_load(datasetA_folder_path)
    train_iter_A = SerialIterator(dataset_A, batch_size, repeat=True, shuffle=True)
    dataset_B = data_io.dataset_load(datasetB_folder_path)
    train_iter_B = SerialIterator(dataset_B, batch_size, repeat=True, shuffle=True)

    g_ab = Generator()
    g_ba = Generator()
    d_a = Discriminator()
    d_b = Discriminator()

    g_ab.to_gpu(0)
    g_ba.to_gpu(0)
    d_a.to_gpu(0)
    d_b.to_gpu(0)

    opt_g_ab = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_g_ab.setup(g_ab)
    opt_g_ba = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_g_ba.setup(g_ba)
    opt_d_a = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_d_a.setup(d_a)
    opt_d_b = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_d_b.setup(d_b)

    iteration = 0
    train_iter_A.reset()
    train_iter_B.reset()

    log_list = []
    image_path = output_path + "image/"
    disA_model_path = output_path + "dis_A/"
    disB_model_path = output_path + "dis_B/"
    genAB_model_path = output_path + "gen_AB/"
    genBA_model_path = output_path + "gen_BA/"
    os.mkdir(output_path)
    os.mkdir(image_path)
    os.mkdir(disA_model_path)
    os.mkdir(disB_model_path)
    os.mkdir(genAB_model_path)
    os.mkdir(genBA_model_path)

    for epoch in range(epoch_count):
        d_a_loss_list = []
        d_b_loss_list = []
        g_AB_loss_list = []
        g_BA_loss_list = []
        while True:
            mini_batch_images_A = train_iter_A.next()
            mini_batch_images_A = np.array(mini_batch_images_A)
            mini_batch_images_A = (mini_batch_images_A - 128.0) / 128.0
            real_a = Variable(np.array(mini_batch_images_A))
            real_a.to_gpu(0)

            mini_batch_images_B = train_iter_B.next()
            mini_batch_images_B = np.array(mini_batch_images_B)
            mini_batch_images_B = (mini_batch_images_B - 128.0) / 128.0
            real_b = Variable(np.array(mini_batch_images_B))
            real_b.to_gpu(0)


            fake_b = g_ab(real_a)
            fake_a = g_ba(real_b)

            reconstr_a = g_ba(fake_b)
            reconstr_b = g_ab(fake_a)

            d_a_real_result = d_a(real_a)
            d_a_fake_result = d_a(fake_a)
            loss_d_a = loss_dis(batch_size, d_a_real_result, d_a_fake_result)

            d_b_real_result = d_b(real_b)
            d_b_fake_result = d_b(fake_b)
            loss_d_b = loss_dis(batch_size, d_b_real_result, d_b_fake_result)

            d_a.cleargrads()
            loss_d_a.backward()
            opt_d_a.update()

            d_b.cleargrads()
            loss_d_b.backward()
            opt_d_b.update()


            """generatorのloss計算"""
            loss_g_ab = loss_gen()
            loss_g_ba = loss_gen()

            g_ab.cleargrads()
            loss_g_ab.backward()
            opt_g_ab.update()

            g_ba.cleargrads()
            loss_g_ba.backward()
            opt_g_ba.update()


            loss_d_a.to_cpu()
            loss_d_b.to_cpu()
            loss_g_ab.to_cpu()
            loss_g_ba.to_cpu()

            iteration += batch_size
            d_a_loss_list.append(loss_d_a.array)
            d_b_loss_list.append(loss_d_b.array)
            g_AB_loss_list.append(loss_g_ab.array)
            g_BA_loss_list.append(loss_g_ba.array)

            if train_iter.is_new_epoch:
                break


        real_a.to_cpu()
        fake_b.to_cpu()
        reconstr_a.to_cpu()
        real_b.to_cpu()
        fake_a.to_cpu()
        reconstr_b.to_cpu()
        real_a_images = real_a.array.transpose(0, 2, 3, 1)
        fake_b_images = fake_b.array.transpose(0, 2, 3, 1)
        reconstr_a_images = reconstr_a.array.transpose(0, 2, 3, 1)
        real_b_images = real_b.array.transpose(0, 2, 3, 1)
        fake_a_images = fake_a.array.transpose(0, 2, 3, 1)
        reconstr_b_images = reconstr_b.array.transpose(0, 2, 3, 1)
        data_io.output_images(real_a_images, fake_b_images, reconstr_a_images, real_b_images, fake_a_images, reconstr_b_images)

        print("epoch: " + str(epoch) + ", interation: " + str(iteration) + \
            ", d_A_loss: " + str(np.mean(d_a_loss_list)) + ", d_B_loss: " + str(np.mean(d_b_loss_list)) + \
            ", g_AB_loss: " + str(np.mean(g_AB_loss_list)) + ", g_BA_loss: " + str(np.mean(g_BA_loss_list)))

        log_json = {"epoch": str(epoch), "interation": str(iteration), \
            "d_A_loss": str(np.mean(d_a_loss_list)), "d_B_loss": str(np.mean(d_b_loss_list)), \
            "g_AB_loss": str(np.mean(g_AB_loss_list)), "g_BA_loss": str(np.mean(g_BA_loss_list))}
        log_list.append(log_json)
        with open(output_path + 'log.json', 'w') as log_file:
            json.dump(log_list, log_file, indent=4)

        if(epoch % 100 == 0):
            g_ab.to_cpu()
            g_ba.to_cpu()
            d_a.to_cpu()
            d_b.to_cpu()
            save_npz(genAB_model_path + str(epoch) + '.npz', g_ab)
            save_npz(genBA_model_path + str(epoch) + '.npz', g_ba)
            save_npz(disA_model_path + str(epoch) + '.npz', d_a)
            save_npz(disB_model_path + str(epoch) + '.npz', d_b)
            g_ab.to_gpu(0)
            g_ba.to_gpu(0)
            d_a.to_gpu(0)
            d_b.to_gpu(0)


    g_ab.to_cpu()
    g_ba.to_cpu()
    d_a.to_cpu()
    d_b.to_cpu()
    save_npz(genAB_model_path + 'last.npz', g_ab)
    save_npz(genBA_model_path + 'last.npz', g_ba)
    save_npz(disA_model_path + 'last.npz', d_a)
    save_npz(disB_model_path + 'last.npz', d_b)


def loss_dis(batchsize, real_result, fake_result):
    return 1


def loss_gen():
    return 1



if __name__ == '__main__':
    now_time = datetime.datetime.now()
    folder_name = "{0:%Y-%m-%d_%H-%M-%S}".format(now_time)

    parser = argparse.ArgumentParser(description='Chainer-GAN: cyclegan')
    parser.add_argument('--batchsize', '-b', type=int, default=55,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--datasetA', '-dA', default='datasetA',
                        help='Directory of dataset image files. Default is /Chainer-GAN/cyclegan/datasetA')
    parser.add_argument('--datasetB', '-dB', default='datasetB',
                        help='Directory of dataset image files. Default is /Chainer-GAN/cyclegan/datasetB')
    parser.add_argument('--out', '-o', default='output/' + folder_name + "/",
                        help='Directory to output the result. Default is /Chainer-GAN/cyclegan/output/yyyy-mm-dd_HH-MM-SS')
    args = parser.parse_args()

    train(args.batchsize, args.epoch, args.datasetA, args.datasetB, args.out)
